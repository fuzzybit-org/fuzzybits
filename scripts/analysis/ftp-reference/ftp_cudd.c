/* ftp_cudd.c: build the BDD of fault-tree targets with CUDD (dynamic reordering) and dump it.
 *
 * Input: the plain ".net" file written by ftp_cudd.py (variables in initial order, gates in
 * topological order, targets). Output: per target, the BDD node list (post-order, children
 * first) to <dump-prefix>.<target>.bdd, plus probabilities in double and long double computed
 * as a (P, 1-P) pair per regular node so complemented edges never subtract.
 *
 * Usage: ftp_cudd NET DUMP_PREFIX [--reorder sift|sift-converge|group-sift|symm-sift|none]
 *                  [--max-mem-mb N] [--final-reduce] [--progress-sec S]
 *
 * Build (CUDD 3.0.0 installed under $CUDD):
 *   cc -O2 -std=c11 -I$CUDD/include ftp_cudd.c -o ftp_cudd $CUDD/lib/libcudd.a -lm
 */
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

#include "cudd.h"

static double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static long peak_rss_mb(void) {
  struct rusage ru;
  getrusage(RUSAGE_SELF, &ru);
  return ru.ru_maxrss / 1024; /* Linux: KiB */
}

typedef struct {
  int kind; /* 0 or, 1 and, 2 atleast */
  int k, nops;
  int *op;  /* encoded: (idx << 2) | (is_gate << 1) | complemented */
} Gate;

static char **vname;
static char **vprob_str;
static int nvars, ngates, ntargets;
static Gate *gates;
static int *targets;
static DdNode **gbdd;
static int *gpending;

static DdNode *operand(DdManager *dd, int code) {
  int idx = code >> 2, isg = (code >> 1) & 1, neg = code & 1;
  DdNode *f = isg ? gbdd[idx] : Cudd_bddIthVar(dd, idx);
  return neg ? Cudd_Not(f) : f;
}

/* --- pointer -> index hash map (open addressing) --- */
typedef struct {
  DdNode **key;
  int *val;
  size_t cap;
} PMap;

static void pm_init(PMap *m, size_t n) {
  m->cap = 16;
  while (m->cap < 2 * n + 16) m->cap <<= 1;
  m->key = calloc(m->cap, sizeof(DdNode *));
  m->val = malloc(m->cap * sizeof(int));
}
static size_t pm_slot(PMap *m, DdNode *p) {
  size_t h = ((uintptr_t)p >> 4) * 0x9E3779B97F4A7C15ull;
  h &= m->cap - 1;
  while (m->key[h] && m->key[h] != p) h = (h + 1) & (m->cap - 1);
  return h;
}

typedef struct {
  int var, then_e, else_e; /* edges: 2*nodeid + complemented; node 0 = constant ONE */
} DNode;

static DNode *nodes;
static int nnodes;
static PMap pm;

static int visit(DdManager *dd, DdNode *r) { /* r regular; returns node id */
  size_t s = pm_slot(&pm, r);
  if (pm.key[s]) return pm.val[s];
  int id;
  if (Cudd_IsConstant(r)) {
    id = 0;
  } else {
    DdNode *t = Cudd_T(r), *e = Cudd_E(r);
    int ti = visit(dd, Cudd_Regular(t));
    int ei = visit(dd, Cudd_Regular(e));
    id = nnodes++;
    nodes[id].var = Cudd_NodeReadIndex(r);
    nodes[id].then_e = 2 * ti + (Cudd_IsComplement(t) ? 1 : 0);
    nodes[id].else_e = 2 * ei + (Cudd_IsComplement(e) ? 1 : 0);
  }
  s = pm_slot(&pm, r);
  pm.key[s] = r;
  pm.val[s] = id;
  return id;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s NET DUMP_PREFIX [options]\n", argv[0]);
    return 2;
  }
  const char *net = argv[1], *prefix = argv[2];
  Cudd_ReorderingType method = CUDD_REORDER_SIFT;
  const char *mname = "sift";
  size_t max_mem_mb = 32768;
  int final_reduce = 0;
  double progress_sec = 10.0;
  for (int i = 3; i < argc; i++) {
    if (!strcmp(argv[i], "--reorder") && i + 1 < argc) {
      mname = argv[++i];
      if (!strcmp(mname, "sift")) method = CUDD_REORDER_SIFT;
      else if (!strcmp(mname, "sift-converge")) method = CUDD_REORDER_SIFT_CONVERGE;
      else if (!strcmp(mname, "group-sift")) method = CUDD_REORDER_GROUP_SIFT;
      else if (!strcmp(mname, "group-sift-converge")) method = CUDD_REORDER_GROUP_SIFT_CONV;
      else if (!strcmp(mname, "symm-sift")) method = CUDD_REORDER_SYMM_SIFT;
      else if (!strcmp(mname, "none")) method = CUDD_REORDER_NONE;
      else { fprintf(stderr, "unknown reorder %s\n", mname); return 2; }
    } else if (!strcmp(argv[i], "--max-mem-mb") && i + 1 < argc) {
      max_mem_mb = strtoul(argv[++i], NULL, 10);
    } else if (!strcmp(argv[i], "--final-reduce")) {
      final_reduce = 1;
    } else if (!strcmp(argv[i], "--progress-sec") && i + 1 < argc) {
      progress_sec = atof(argv[++i]);
    } else {
      fprintf(stderr, "unknown option %s\n", argv[i]);
      return 2;
    }
  }

  FILE *in = fopen(net, "r");
  if (!in) { perror(net); return 1; }
  char tag[64];
  if (fscanf(in, "%63s %d", tag, &nvars) != 2 || strcmp(tag, "vars")) { fprintf(stderr, "bad net\n"); return 1; }
  vname = malloc(nvars * sizeof(char *));
  vprob_str = malloc(nvars * sizeof(char *));
  for (int i = 0; i < nvars; i++) {
    char nb[4096], pb[256];
    if (fscanf(in, "%4095s %255s", nb, pb) != 2) { fprintf(stderr, "bad var\n"); return 1; }
    vname[i] = strdup(nb);
    vprob_str[i] = strdup(pb);
  }
  if (fscanf(in, "%63s %d", tag, &ngates) != 2 || strcmp(tag, "gates")) { fprintf(stderr, "bad net\n"); return 1; }
  gates = calloc(ngates, sizeof(Gate));
  char **gname = malloc(ngates * sizeof(char *));
  for (int g = 0; g < ngates; g++) {
    char nb[4096], kb[16];
    if (fscanf(in, "%4095s %15s %d %d", nb, kb, &gates[g].k, &gates[g].nops) != 4) { fprintf(stderr, "bad gate\n"); return 1; }
    gname[g] = strdup(nb);
    gates[g].kind = !strcmp(kb, "or") ? 0 : !strcmp(kb, "and") ? 1 : 2;
    gates[g].op = malloc(gates[g].nops * sizeof(int));
    for (int j = 0; j < gates[g].nops; j++) {
      char ob[64];
      if (fscanf(in, "%63s", ob) != 1) { fprintf(stderr, "bad op\n"); return 1; }
      char *p = ob;
      int neg = 0;
      if (*p == '-') { neg = 1; p++; }
      int isg = (*p == 'g');
      int idx = atoi(p + 1);
      if (isg && idx >= g) { fprintf(stderr, "gates not topological\n"); return 1; }
      gates[g].op[j] = (idx << 2) | (isg << 1) | neg;
    }
  }
  if (fscanf(in, "%63s %d", tag, &ntargets) != 2 || strcmp(tag, "targets")) { fprintf(stderr, "bad net\n"); return 1; }
  targets = malloc(ntargets * sizeof(int));
  for (int t = 0; t < ntargets; t++)
    if (fscanf(in, "%d", &targets[t]) != 1) { fprintf(stderr, "bad target\n"); return 1; }
  fclose(in);

  /* remaining-parent counts; targets get +1 so they survive */
  gpending = calloc(ngates, sizeof(int));
  for (int g = 0; g < ngates; g++)
    for (int j = 0; j < gates[g].nops; j++)
      if ((gates[g].op[j] >> 1) & 1) gpending[gates[g].op[j] >> 2]++;
  for (int t = 0; t < ntargets; t++) gpending[targets[t]]++;

  DdManager *dd = Cudd_Init(nvars, 0, CUDD_UNIQUE_SLOTS, CUDD_CACHE_SLOTS, (size_t)max_mem_mb << 20);
  Cudd_SetMaxMemory(dd, (size_t)max_mem_mb << 20);
  if (method != CUDD_REORDER_NONE) Cudd_AutodynEnable(dd, method);
  else Cudd_AutodynDisable(dd);

  double t0 = now(), last = t0;
  gbdd = calloc(ngates, sizeof(DdNode *));
  for (int g = 0; g < ngates; g++) {
    Gate *G = &gates[g];
    DdNode *f;
    if (G->kind == 0 || G->kind == 1) {
      f = G->kind == 0 ? Cudd_ReadLogicZero(dd) : Cudd_ReadOne(dd);
      Cudd_Ref(f);
      for (int j = 0; j < G->nops; j++) {
        DdNode *x = operand(dd, G->op[j]);
        DdNode *h = G->kind == 0 ? Cudd_bddOr(dd, f, x) : Cudd_bddAnd(dd, f, x);
        if (!h) { fprintf(stderr, "CUDD failure (memory/limit) in gate %s; error %d\n", gname[g], Cudd_ReadErrorCode(dd)); goto fail; }
        Cudd_Ref(h);
        Cudd_RecursiveDeref(dd, f);
        f = h;
      }
    } else {
      /* at-least-k DP: a[j] = "at least j of the operands seen so far", a[0] = 1 */
      int k = G->k;
      DdNode **a = malloc((k + 1) * sizeof(DdNode *));
      a[0] = Cudd_ReadOne(dd);
      Cudd_Ref(a[0]);
      for (int j = 1; j <= k; j++) { a[j] = Cudd_ReadLogicZero(dd); Cudd_Ref(a[j]); }
      for (int i = 0; i < G->nops; i++) {
        DdNode *x = operand(dd, G->op[i]);
        for (int j = k; j >= 1; j--) {
          DdNode *h = Cudd_bddIte(dd, x, a[j - 1], a[j]);
          if (!h) { fprintf(stderr, "CUDD failure in atleast gate %s\n", gname[g]); goto fail; }
          Cudd_Ref(h);
          Cudd_RecursiveDeref(dd, a[j]);
          a[j] = h;
        }
      }
      f = a[k];
      for (int j = 0; j < k; j++) Cudd_RecursiveDeref(dd, a[j]);
      free(a);
    }
    gbdd[g] = f;
    for (int j = 0; j < G->nops; j++)
      if ((G->op[j] >> 1) & 1) {
        int c = G->op[j] >> 2;
        if (--gpending[c] == 0) { Cudd_RecursiveDeref(dd, gbdd[c]); gbdd[c] = NULL; }
      }
    double tn = now();
    if (tn - last >= progress_sec || g == ngates - 1) {
      last = tn;
      fprintf(stderr, "[%.1fs] gate %d/%d %s: size %d, live nodes %ld, peak %ld, reorderings %d, mem %.0f MB, rss %ld MB\n",
              tn - t0, g + 1, ngates, gname[g], Cudd_DagSize(f), Cudd_ReadNodeCount(dd), Cudd_ReadPeakNodeCount(dd),
              Cudd_ReadReorderings(dd), Cudd_ReadMemoryInUse(dd) / 1048576.0, peak_rss_mb());
    }
  }
  double tbuild = now() - t0;
  int reord_build = Cudd_ReadReorderings(dd);
  long reord_ms_build = Cudd_ReadReorderingTime(dd);
  printf("build_seconds %.3f\nreorder_method %s\nreorderings_during_build %d\nreorder_ms_during_build %ld\n", tbuild, mname,
         reord_build, reord_ms_build);
  for (int t = 0; t < ntargets; t++)
    printf("target %s size_after_build %d\n", gname[targets[t]], Cudd_DagSize(gbdd[targets[t]]));
  if (final_reduce) {
    double tr = now();
    Cudd_ReduceHeap(dd, CUDD_REORDER_SIFT_CONVERGE, 0);
    printf("final_reduce_seconds %.3f\n", now() - tr);
  }
  printf("reorderings_total %d\nreorder_ms_total %ld\npeak_live_nodes %ld\nmemory_in_use_mb %.1f\n", Cudd_ReadReorderings(dd),
         Cudd_ReadReorderingTime(dd), Cudd_ReadPeakNodeCount(dd), Cudd_ReadMemoryInUse(dd) / 1048576.0);

  long double *lp = malloc(nvars * sizeof(long double));
  for (int i = 0; i < nvars; i++) lp[i] = strtold(vprob_str[i], NULL);

  for (int t = 0; t < ntargets; t++) {
    DdNode *f = gbdd[targets[t]];
    int sz = Cudd_DagSize(f);
    nodes = malloc((sz + 1) * sizeof(DNode));
    nnodes = 1;
    pm_init(&pm, sz + 1);
    int root = visit(dd, Cudd_Regular(f));
    int root_e = 2 * root + (Cudd_IsComplement(f) ? 1 : 0);
    /* (P, Pbar) per regular node, double and long double; only additions of non-negative terms */
    double *P = malloc(nnodes * sizeof(double)), *Q = malloc(nnodes * sizeof(double));
    long double *LP = malloc(nnodes * sizeof(long double)), *LQ = malloc(nnodes * sizeof(long double));
    P[0] = 1; Q[0] = 0; LP[0] = 1; LQ[0] = 0;
    for (int n = 1; n < nnodes; n++) {
      int te = nodes[n].then_e, ee = nodes[n].else_e;
      long double p = lp[nodes[n].var], q = 1.0L - p;
      double pd = (double)p, qd = (double)q;
      double pt = (te & 1) ? Q[te >> 1] : P[te >> 1], qt = (te & 1) ? P[te >> 1] : Q[te >> 1];
      double pe = (ee & 1) ? Q[ee >> 1] : P[ee >> 1], qe = (ee & 1) ? P[ee >> 1] : Q[ee >> 1];
      P[n] = pd * pt + qd * pe;
      Q[n] = pd * qt + qd * qe;
      long double lpt = (te & 1) ? LQ[te >> 1] : LP[te >> 1], lqt = (te & 1) ? LP[te >> 1] : LQ[te >> 1];
      long double lpe = (ee & 1) ? LQ[ee >> 1] : LP[ee >> 1], lqe = (ee & 1) ? LP[ee >> 1] : LQ[ee >> 1];
      LP[n] = p * lpt + q * lpe;
      LQ[n] = p * lqt + q * lqe;
    }
    double pr = (root_e & 1) ? Q[root] : P[root];
    long double lpr = (root_e & 1) ? LQ[root] : LP[root];
    printf("target %s nodes %d (dag size incl. constant %d) prob_double %.17g prob_longdouble %.21Lg\n", gname[targets[t]],
           nnodes - 1, sz, pr, lpr);

    char path[8192];
    snprintf(path, sizeof path, "%s.%s.bdd", prefix, gname[targets[t]]);
    FILE *out = fopen(path, "w");
    if (!out) { perror(path); return 1; }
    /* levels in the final order, so the exact evaluator can scale denominators */
    fprintf(out, "nvars %d\nperm", nvars);
    for (int i = 0; i < nvars; i++) fprintf(out, " %d", Cudd_ReadPerm(dd, i));
    fprintf(out, "\nnodes %d root %d\n", nnodes - 1, root_e);
    for (int n = 1; n < nnodes; n++) fprintf(out, "%d %d %d\n", nodes[n].var, nodes[n].then_e, nodes[n].else_e);
    fclose(out);
    printf("dump %s\n", path);
    free(P); free(Q); free(LP); free(LQ); free(nodes); free(pm.key); free(pm.val);
  }
  printf("final_order");
  for (int l = 0; l < nvars; l++) printf(" %s", vname[Cudd_ReadInvPerm(dd, l)]);
  printf("\npeak_rss_mb %ld\ntotal_seconds %.3f\n", peak_rss_mb(), now() - t0);
  return 0;
fail:
  printf("FAILED after %.1fs: live nodes %ld, peak live nodes %ld, reorderings %d, mem %.0f MB, rss %ld MB\n", now() - t0,
         Cudd_ReadNodeCount(dd), Cudd_ReadPeakNodeCount(dd), Cudd_ReadReorderings(dd), Cudd_ReadMemoryInUse(dd) / 1048576.0,
         peak_rss_mb());
  return 3;
}
