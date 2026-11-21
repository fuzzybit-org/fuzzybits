# RFC: monotone-hull (delete-term) probability queries

Status: proposed  
Issue: [#174](https://github.com/fuzzybit-org/fuzzybits/issues/174) · tracking: [#185](https://github.com/fuzzybit-org/fuzzybits/issues/185)  
Related: [#177](https://github.com/fuzzybit-org/fuzzybits/issues/177) (pass semantic classes),
[#178](https://github.com/fuzzybit-org/fuzzybits/issues/178) (per-assignment check),
[#179](https://github.com/fuzzybit-org/fuzzybits/issues/179) (reference estimators, fixtures),
[#181](https://github.com/fuzzybit-org/fuzzybits/issues/181) (quantification config),
[#183](https://github.com/fuzzybit-org/fuzzybits/issues/183) (runtime),
[#6](https://github.com/fuzzybit-org/fuzzybits/issues/6) (importance sampling)  
Format: `.pcfb` `QueryKind.MONOTONE_HULL_PROBABILITY`, pcfb 1.2
([`SPEC-v1.2.md`](https://github.com/fuzzybit-org/pcfb/blob/main/docs/SPEC-v1.2.md),
[pcfb#10](https://github.com/fuzzybit-org/pcfb/pull/10))

## 0. Summary

PRA models built for cut-set tools write event-tree success branches and delete terms with negation, but they mean
cut-set deletion, not Boolean complement. This RFC defines the monotone hull $\uparrow g$ of a target $g$, proves
that its minimal cut sets are exactly the cut sets a delete-term tool keeps, gives an unbiased Monte Carlo estimator of
$P(\uparrow g)$ that runs an exact check only on samples where two cheap bounds disagree, and fixes the interfaces.

Decisions proposed here:

1. **A query kind, not a rewrite.** `MONOTONE_HULL_PROBABILITY` is a second probability query kind (pcfb 1.2). No
   polynomial-time compiler can turn a circuit for $g$ into a circuit for $\uparrow g$ unless P = NP (§6.1).
2. **Bounds are derived by the runtime**, not emitted by the writer (§7.4): a lower bound $L$ (negative-unate events
   set to 0) and an upper bound $U$ (negation-normal relaxation), both correct by construction.
3. **pcfb 1.2**: one enum value appended; `schema_version` `0x00010002` only when the kind is used; opset unchanged;
   runtimes reject kinds they do not implement before sampling (§7.3).
4. **Transforms** state which query kinds they preserve. Hull targets accept function-preserving passes and folding of
   probability-0 events, not folding of probability-1 events (§7.5).
5. **`ir::pc` is unchanged**; the kind is chosen per target in the `ir::pra` quantification config (§7.1, §7.2).

## 1. Motivation

A private 88k-gate plant model ([#185](https://github.com/fuzzybit-org/fuzzybits/issues/185)) encodes each
sequence as `SEQ = F and not (S or DEL)`, where `DEL` is a delete-term gate: pairwise exclusions among events, plus
combinations involving configuration flags with $p = 1$. Aggregate facts:

- Under literal Boolean semantics the top event is identically FALSE: one `DEL` conjunction consists of two $p = 1$
  flags, so `not DEL` is FALSE in every world.
- With that conjunction forced FALSE, `not DEL` is still FALSE in about 97% of sampled worlds, because `DEL` holds
  pairwise exclusions among a few high-probability events. The Boolean answer is dominated by this artifact.
- The model's intended meaning is cut-set deletion. Open-PSA MEF 2.0 §6.2 describes delete terms as groups of pairwise
  exclusive events and notes that most tools apply them as post-processing (a minimal cut set containing two events of
  a group is discarded), while the logical encoding rewrites the top event as $top \wedge \neg(\text{2-of-group})$
  [MEF]. SAPHIRE's default treatment of success branches ("delete term") removes failed-system cut sets that contain a
  success-system cut set [KSC22].
- A NumPy prototype of the estimator in §5 checked 2,000 lanes: residual problems had a median of 12 free variables,
  each check took 0.21 s in Python, about 10% of lanes with $g_{coh} = 1$ satisfied the hull, and about 3.6% of all
  samples needed a check.

## 2. Setting and notation

Assumptions, used throughout:

- **A1 (finite).** A finite set $V$ of basic events, $n = |V|$. A world is $x \in \lbrace 0,1 \rbrace^V$; $T(x)$ is
  its set of true events, and $1_S$ is the world whose true set is $S$.
- **A2 (independence).** $X = (X_v : v \in V)$ has independent components with $P(X_v = 1) = p_v \in [0, 1]$. This
  is the `.pcfb` `SAMPLE` model (pcfb SPEC §4).
- **A3 (constants are not events).** Constants (house events after resolution, config overrides, `BITWISE_ZEROS` /
  `BITWISE_ONES`) are not elements of $V$. An event with $p_v \in \lbrace 0, 1 \rbrace$ is still an event.

Order: $y \le x$ iff $y_v \le x_v$ for all $v$, i.e. $T(y) \subseteq T(x)$. A Boolean function
$g : \lbrace 0,1 \rbrace^V \to \lbrace 0,1 \rbrace$ is *monotone* if $y \le x$ implies $g(y) \le g(x)$. Reliability
texts call a monotone function in which every variable is relevant *coherent*; only monotonicity is used here.

A *cut set* of $g$ is a set $S$ with $g(1_S) = 1$; it is *minimal* if no proper subset is a cut set;
$\operatorname{mcs}(g)$ is the family of minimal cut sets. A monotone $g$ equals
$\bigvee_{C \in \operatorname{mcs}(g)} \bigwedge_{v \in C} x_v$.

A *literal* is $x_v$ or $\neg x_v$; a *term* $t$ is a set of literals containing no complementary pair, read as a
conjunction; $\operatorname{pos}(t)$ is the set of variables occurring positively in $t$. $t$ is an *implicant* of $g$
if every world satisfying $t$ satisfies $g$; it is *prime* if no proper subterm is an implicant. $\operatorname{PI}(g)$
is the set of prime implicants. For a family $\mathcal{S}$ of sets, $\operatorname{Min}\mathcal{S}$ is its set of
inclusion-minimal members.

Circuits are DAGs over $V$ and constants with the `ir::pc` / `.pcfb` connectives: AND, OR, XOR, NOT, NAND, NOR, XNOR,
cardinality (`ATLEAST`, `ATMOST`, `EXACTLY`, or `cardinality(low, high)`), and complemented edges (`edge_ref.complement`,
pcfb `inverted_inputs`). A target computes a function $g$; its *cone* is the set of nodes it depends on.

Restriction and quantification: $g_{v \leftarrow c}(x) = g(x[v \leftarrow c])$ and $\exists_v g = g_{v \leftarrow 0} \vee g_{v \leftarrow 1}$.

## 3. Definition

$$\uparrow g(x) = \bigvee_{y \le x} g(y).$$

$\uparrow g(x) = 1$ iff some subset of the events that are true in $x$, with every other event false, satisfies $g$.
The query value is $P(\uparrow g(X))$.

In Rauzy's terms, $\uparrow g$ is the widening $\omega_L(g)$ with $L$ the set of positive literals: a world is added to
$g$ when it is "more significant" (has more failures) than a world already in $g$ [R01, §IV.D].

## 4. Properties

### 4.1 Least monotone majorant

**Theorem 1.** (a) $\uparrow g$ is monotone and $g \le \uparrow g$. (b) If $h$ is monotone and $g \le h$, then
$\uparrow g \le h$. (c) $\uparrow$ is a closure operator: $g \le g'$ implies $\uparrow g \le \uparrow g'$;
$\uparrow\uparrow g = \uparrow g$; $\uparrow g = g$ iff $g$ is monotone.

*Proof.* (a) If $x \le x'$, every $y \le x$ satisfies $y \le x'$, so $\uparrow g(x) \le \uparrow g(x')$; taking $y = x$
gives $g \le \uparrow g$. (b) If $\uparrow g(x) = 1$ there is $y \le x$ with $g(y) = 1$, so $h(y) = 1$ and, $h$ being
monotone, $h(x) = 1$. (c) Monotonicity of $\uparrow$ is immediate from the definition. By (a) $\uparrow g$ is a
monotone majorant of itself, so (b) with $h = \uparrow g$ gives $\uparrow\uparrow g \le \uparrow g$, and (a) gives the
reverse. If $g$ is monotone, (b) with $h = g$ gives $\uparrow g \le g$; conversely $\uparrow g$ is monotone. ∎

### 4.2 Minimal cut sets

**Theorem 2.** $\operatorname{mcs}(\uparrow g) = \operatorname{Min}\lbrace \operatorname{pos}(t) : t \in \operatorname{PI}(g) \rbrace = \operatorname{Min}\lbrace S : g(1_S) = 1 \rbrace$.

The minimal cut sets of the hull are the minimal positive parts of the prime implicants of $g$, which is the usual
definition of minimal cut sets of a non-coherent function. This is Rauzy's $MC_L[F]$ with $L$ the positive literals
[R01, §IV.C]: his Theorem 1 states $PI_L[F] = MC_L[F]$ and his Theorem 2 states $PI[\omega_L(F)] = MC_L[F]$ (theorem
numbers of the author's preprint); with $L$ the positive literals these are the *minimal p-cuts* of [DR97]. A
self-contained proof follows.

*Proof.* Let $G = \lbrace S : g(1_S) = 1 \rbrace$ and $P = \lbrace \operatorname{pos}(t) : t \in \operatorname{PI}(g) \rbrace$.

1. $\uparrow g(1_S) = 1$ iff some $S' \subseteq S$ lies in $G$. So the cut sets of $\uparrow g$ are the up-closure of
   $G$, and $\operatorname{mcs}(\uparrow g) = \operatorname{Min} G$.
2. $P \subseteq G$: for an implicant $t$, the world $1_{\operatorname{pos}(t)}$ makes every positive literal of $t$ true
   and, since $t$ has no complementary pair, every negative literal of $t$ true. So it satisfies $t$, hence $g$.
3. Every $S \in G$ contains a member of $P$: the minterm of $1_S$ is an implicant; deleting literals while the term
   stays an implicant ends in a prime implicant $t$ whose literals are a subset of the minterm's, so
   $\operatorname{pos}(t) \subseteq S$.
4. Let $m \in \operatorname{Min} G$. By 3 some $p \in P$ has $p \subseteq m$; by 2 $p \in G$, so $p = m$ by minimality,
   and $m$ is minimal in $P$ because $P \subseteq G$. Conversely let $p \in \operatorname{Min} P$; $p \in G$ by 2; if
   some $S \in G$ had $S \subsetneq p$, then by 3 some $p' \in P$ would have $p' \subseteq S \subsetneq p$. So
   $p \in \operatorname{Min} G$. ∎

Example [R01]: $g = ab \vee \neg a c$ has prime implicants $ab$, $\neg a c$, $bc$; the minimal positive parts are $ab$
and $c$, so $\uparrow g = ab \vee c$.

### 4.3 Delete terms

**Theorem 3.** Let $F$ and $D$ be monotone and $g = F \wedge \neg D$. Let
$K = \lbrace C \in \operatorname{mcs}(F) : \text{no } C' \in \operatorname{mcs}(D) \text{ has } C' \subseteq C \rbrace$.
Then $\uparrow g = \bigvee_{C \in K} \bigwedge_{v \in C} x_v$ and $\operatorname{mcs}(\uparrow g) = K$.

*Proof.* For $C \in \operatorname{mcs}(F)$, $D(1_C) = 0$ iff $C$ contains no minimal cut set of $D$. If $C \in K$ and
$C \subseteq T(x)$, then $y = 1_C \le x$ has $F(y) = 1$, $D(y) = 0$, so $\uparrow g(x) = 1$. Conversely, if
$y \le x$ has $F(y) = 1$ and $D(y) = 0$, pick $C \in \operatorname{mcs}(F)$ with $C \subseteq T(y)$; then
$D(1_C) \le D(y) = 0$ by monotonicity, so $C \in K$ and $C \subseteq T(x)$. $K$ is a subfamily of the antichain
$\operatorname{mcs}(F)$, hence an antichain, hence the minimal cut set family of the monotone function it defines. ∎

**Corollary 3.1 (sequences).** For $SEQ = F \wedge \neg(S_1 \vee \dots \vee S_m \vee DEL)$ with monotone $F$, $S_i$,
$DEL$, the hull keeps exactly the minimal cut sets of $F$ that contain no minimal cut set of any success system $S_i$
and no minimal cut set of $DEL$. For a delete-term group $\lbrace e_1, \dots, e_k \rbrace$ written as
$DEL = \mathrm{atleast}(2;\ e_1, \dots, e_k)$, whose minimal cut sets are the pairs, this is the MEF post-processing rule.
The Boolean value $P(F \wedge \neg D) = P(F) - P(F \wedge D)$ instead removes every world in which $D$ holds, including
worlds where $D$'s events occur next to a cut set of $F$ that does not contain them.

Theorem 3 reproduces the published SAPHIRE example of [KSC22] (sequence 4). Write $A, \dots, G$ for their basic events
`BE-A` … `BE-G`, with `BE-F` renamed $F'$ to avoid a clash. The failed systems give
$F = IE \wedge (C \vee B E) \wedge (F' \vee D G)$ and the success system is $S = A \vee C \vee B D$. Of the four cut sets
of $F$ only $\lbrace IE, B, E, F' \rbrace$ contains no cut set of $S$, which is their delete-term result; the single
prime implicant obtained with negations kept (their "I" process flag),
$IE \wedge B \wedge E \wedge F' \wedge \neg A \wedge \neg C \wedge \neg D$, has exactly that positive part (Theorem 2).

The hull reproduces delete-term processing including its known conservatism: a success branch removes only cut sets
that contain a success-system cut set and is not multiplied by $P(\neg S)$ (§8.5; [KSC22] discusses the resulting
overestimate when success probabilities are not close to 1). Theorem 3 needs $F$ and $D$ monotone; Theorem 2 defines
the hull for any $g$.

### 4.4 Bounds

Three functions, each computable by one bit-parallel pass over a circuit:

- **Edge relaxation $g_{coh}$.** Replace every complemented edge by TRUE, every NOT, NAND, NOR, XNOR and `ATMOST` by
  TRUE, XOR by OR, `EXACTLY k` by `ATLEAST k`, and `cardinality(low, high)` by `ATLEAST low`. This is the bound named
  in #174 ("negated inputs replaced by TRUE").
- **Negation-normal relaxation $U$.** Give every node $v$ a positive rail $v^+$ (computing $v$) and a negative rail
  $v^-$ (computing $\neg v$), built only from AND, OR and `ATLEAST`: for AND, $v^+ = \bigwedge \ell_j$ and
  $v^- = \bigvee \bar\ell_j$, where $\ell_j$ is the child's positive rail (negative rail if the edge is complemented)
  and $\bar\ell_j$ the opposite rail; OR dually; NOT swaps rails; `ATLEAST k` of $m$ has
  $v^+ = [\sum_j \ell_j \ge k]$, $v^- = [\sum_j \bar\ell_j \ge m-k+1]$; NAND, NOR, XNOR, `ATMOST`, `EXACTLY` and
  `cardinality(low, high)` follow by De Morgan; XOR uses the parity chain $P_j = P_{j-1}\bar\ell_j \vee N_{j-1}\ell_j$,
  $N_j = P_{j-1}\ell_j \vee N_{j-1}\bar\ell_j$. Leaves: $x_v^+ = x_v$, $x_v^- = z_v$. Then
  $g(x) = G(x, \neg x)$ for the monotone function $G(x, z)$ computed by the target's positive rail, and
  $U(x) = G(x, 1)$: every negative literal relaxed to TRUE. A node whose cone contains no negation has $v^+ = v$ under
  the relaxation, and its negative rail is the constant $\neg v(0)$, so $U$ costs extra only where negations reach.
- **Negative-unate zeroing $L$.** Call $v$ *negative-unate* if leaf $x_v$ does not occur in the target's positive rail
  (only $z_v$ does), *positive-unate* if $z_v$ does not occur, *binate* if both occur. With $m^-$ the set of
  negative-unate events, $L(x) = g(x \wedge \neg 1_{m^-})$: evaluate $g$ with negative-unate events forced to 0.

**Theorem 4.** Pointwise, $g \le L \le \uparrow g \le U \le g_{coh}$; $U$ and $g_{coh}$ are monotone.

*Proof.*

- Unateness: $G$ is monotone in $(x, z)$. If $v$ is negative-unate, $G$ does not depend on $x_v$, so
  $g(x) = G(x, \neg x)$ is antitone in $x_v$; symmetrically for positive-unate.
- $g \le L$: lowering antitone coordinates cannot decrease $g$.
- $L \le \uparrow g$: $x \wedge \neg 1_{m^-} \le x$.
- $\uparrow g \le U$: for $y \le x$, $g(y) = G(y, \neg y) \le G(x, 1) = U(x)$ by monotonicity of $G$; $U$ is monotone
  because $G$ is.
- $U \le g_{coh}$: by induction in topological order, $v^+(x, 1) \le v_{coh}(x)$ for every node. For a complemented
  edge $g_{coh}$ uses TRUE where $U$ uses a rail value; for NOT, NAND, NOR, XNOR, `ATMOST` $g_{coh}$ is TRUE; for XOR,
  $P_j \le P_{j-1} \vee \ell_j$ gives $P_m \le \bigvee_j \ell_j$ by induction on the chain; `EXACTLY` and
  `cardinality` drop a conjunct; AND, OR, `ATLEAST` are monotone in their inputs, which are dominated by the induction
  hypothesis. $g_{coh}$ is a composition of monotone gates. ∎

Remarks:

- If the cone has no negation at all, $g = L = \uparrow g = U = g_{coh}$. For Theorem 3's form with $F$ and $D$ written
  without negations, $U = g_{coh} = F$.
- $L$, $U$ and $g_{coh}$ depend on the circuit, not only on $g$ ($A \wedge \neg\neg B$ has $g_{coh} = A$,
  $U = A \wedge B$). The estimator's correctness needs only the inequalities; tightness affects cost only.
- In the two-flag example (§8.2) the flags occur only under negation, so they are negative-unate and $L = \uparrow g$;
  whether this holds on a given model is a measurement, not a theorem.

### 4.5 Probability-0 and probability-1 events

**Theorem 5.** Let $h = \uparrow g$. (a) $h_{v \leftarrow 0} = \uparrow(g_{v \leftarrow 0})$. (b) $h_{v \leftarrow 1} = \uparrow(\exists_v g) = \uparrow(g_{v \leftarrow 0}) \vee \uparrow(g_{v \leftarrow 1})$.

*Proof.* (a) $y \le x[v \leftarrow 0]$ forces $y_v = 0$, and $\lbrace y[v \leftarrow 0] : y \le x \rbrace = \lbrace y \le x[v \leftarrow 0] \rbrace$.
(b) $y \le x[v \leftarrow 1]$ leaves $y_v$ free; splitting on $y_v$ gives the two hulls; the middle equality is
Theorem 6(a). ∎

**Corollary 5.1.** If $p_v = 0$ then $X = X[v \leftarrow 0]$ almost surely and $P(\uparrow g) = P(\uparrow(g_{v \leftarrow 0}))$:
replacing the event by the constant FALSE preserves the hull probability. If $p_v = 1$ then
$P(\uparrow g) = P(\uparrow(\exists_v g))$, which differs from $P(\uparrow(g_{v \leftarrow 1}))$ whenever
$\uparrow(g_{v \leftarrow 0}) \not\le \uparrow(g_{v \leftarrow 1})$ on a set of positive probability. Substituting TRUE for a probability-1
event is wrong for the hull; the correct elimination is existential quantification: the event may be dropped.

Under Boolean semantics probability-1 events behave as constants; under the hull they are droppable, and only
probability-0 events behave as constants. This matches cut-set tools, which keep probability-1 events in cut sets and
apply delete terms to them. Constants (A3) are not droppable: a house event set TRUE or an override `X := TRUE` stays
TRUE in every $y$.

### 4.6 OR, AND and modules

**Theorem 6.** (a) $\uparrow(g \vee h) = \uparrow g \vee \uparrow h$. (b) $\uparrow(g \wedge h) \le \uparrow g \wedge \uparrow h$,
with equality when $g$ and $h$ depend on disjoint sets of events. (c) Equality in (b) fails in general, even when $h$ is
monotone: $g = \neg A$, $h = A$ give $\uparrow(g \wedge h) = 0$ and $\uparrow g \wedge \uparrow h = A$.

*Proof.* (a) $\exists y \le x$ distributes over $\vee$. (b) A witness $y$ for $g \wedge h$ witnesses both. With
disjoint supports $V_g$, $V_h$, witnesses $y_g \le x$ on $V_g$ and $y_h \le x$ on $V_h$ combine into one $y \le x$. ∎

Consequences:

- Per-lane hull indicators compose by OR on the same samples: the hull of an end state or accident class (an OR of
  sequences) is the OR of the per-sequence indicators, at no extra check cost.
- Hulls of mutually exclusive sequences are in general **not** mutually exclusive:
  $P(\uparrow(s_1 \vee s_2)) \le P(\uparrow s_1) + P(\uparrow s_2)$, strictly in §8.5. End-state values are their own
  queries; do not sum per-sequence hull values.
- There is no decomposition through AND: $\uparrow(F \wedge \neg D) \ne F \wedge \uparrow\neg D$, and
  $\uparrow\neg D = 1$ whenever $D(0) = 0$.

**Proposition 6.1 (modules).** Let $g(x) = G(M(x_A), x_B)$ with $A \cap B = \emptyset$ and $M(0) = 0$. Then
$P(\uparrow g) = P(\uparrow G(Z, X_B))$, where $Z$ is an independent Bernoulli variable with $P(Z = 1) = P(\uparrow M)$.

*Proof.* For $y_A \le x_A$ the values of $M(y_A)$ include 0 (take $y_A = 0$) and include 1 iff $\uparrow M(x_A) = 1$.
Hence $\uparrow g(x) = \exists y_B \le x_B : G(0, y_B) \vee (\uparrow M(x_A) \wedge G(1, y_B)) = \uparrow G(\uparrow M(x_A), x_B)$,
and $\uparrow M(X_A)$ is independent of $X_B$. ∎

Collapsing a module into one `COLLAPSED_MODULE` sample with its Boolean probability $P(M)$ is therefore correct for a
hull target only if $P(\uparrow M) = P(M)$, for example when $M$ is monotone (§8.7).

### 4.7 Relation to cut-set quantification and Boolean probability

By Theorem 4 and monotonicity of $P$: $P(g) \le P(L) \le P(\uparrow g) \le P(U) \le P(g_{coh})$. The gap between
$P(g)$ and $P(\uparrow g)$ is unbounded in ratio: in the private model $P(g) = 0 < P(\uparrow g)$ (§8.2 in
miniature).

A cut-set tool with delete terms computes (a truncated part of) $K = \operatorname{mcs}(\uparrow g)$ (Theorem 3) and
quantifies it with the rare-event approximation $\mathrm{REA} = \sum_{C \in K} P(C)$ or the min-cut upper bound
$\mathrm{MCUB} = 1 - \prod_{C \in K}(1 - P(C))$, where $P(C) = \prod_{v \in C} p_v$.

**Proposition 7.** Under A2, without truncation, $P(\uparrow g) \le \mathrm{MCUB} \le \mathrm{REA}$.

*Proof.* The events $\lbrace C \subseteq T(X) \rbrace$ are increasing in $X$; their complements are decreasing.
Independent variables are associated, and decreasing functions of associated variables are positively correlated
[EPW67], so $P(\bigcap_C \lbrace C \not\subseteq T(X) \rbrace) \ge \prod_C (1 - P(C))$; this is the Esary–Proschan
min-cut bound [EP63]. $1 - \prod(1 - a_i) \le \sum a_i$ for $a_i \in [0, 1]$ by induction. ∎

REA can exceed 1 when probability-1 events occur in cut sets (§8.1).

**Truncation.** Tools drop cut sets below a probability cutoff (or above a size). If cut sets of $D$ are truncated at
$\tau_D$ and cut sets of $F$ at $\tau_F$ with $\tau_D \le \tau_F$, no surviving $C \in \operatorname{mcs}(F)$ escapes
deletion: a truncated $C' \in \operatorname{mcs}(D)$ with $C' \subseteq C$ has $P(C) \le P(C') < \tau_D \le \tau_F$, so
$C$ is truncated too (for size cutoffs the condition is that $D$'s size limit is at least $F$'s, since
$|C| \ge |C'|$). The tool then quantifies $K_\tau = \lbrace C \in K : P(C) \ge \tau_F \rbrace$, and
$P(\uparrow g) - P(\exists C \in K_\tau : C \subseteq T(X)) \le \sum_{C \in K \setminus K_\tau} P(C)$. If
$\tau_D > \tau_F$, deletions can be missed. A tool's number is therefore neither an upper nor a lower bound on $P(\uparrow g)$ in general: truncation
lowers it, REA and MCUB raise it. Generating cut sets with negations dropped (success branches ignored) gives
$\operatorname{mcs}(U)$ and overestimates $P(\uparrow g)$ (§8.4).

**Complexity.** For monotone $g$, $\uparrow g = g$, so computing $P(\uparrow g)$ exactly includes monotone network
reliability, which is #P-hard [V79]. At scale the choice is between sampling with a stated interval and cut-set
approximations of unknown sign; this RFC specifies the former.

## 5. Estimator

### 5.1 Per-lane evaluation

Draw $N$ independent worlds ("lanes") $X_1, \dots, X_N$. For each lane evaluate $L$ and $U$ bit-parallel (64 lanes per
word, or any width), then set

- $H_k = 0$ if $U(X_k) = 0$;
- $H_k = 1$ if $L(X_k) = 1$;
- otherwise ($L = 0$, $U = 1$: a *candidate* lane) $H_k = \uparrow g(X_k)$, decided exactly by the check in §6.

The estimate is $\hat\theta = N^{-1}\sum_k H_k$. Using $g$ and $g_{coh}$ in place of $L$ and $U$ is also correct; $L$ and
$U$ only reduce the number of candidates. The device produces a candidate mask per word; the host gathers the event
bits of candidate lanes, runs the checks in parallel, and adds integer counts.

### 5.2 Unbiasedness, variance, intervals

By Theorem 4, $H_k = \uparrow g(X_k)$ on every lane, so the $H_k$ are i.i.d. Bernoulli($\theta$) with
$\theta = P(\uparrow g)$: $E\hat\theta = \theta$ and $\operatorname{Var}\hat\theta = \theta(1 - \theta)/N$.
Intervals: Wilson score [W27] by default; Clopper–Pearson [CP34] when hits are few. With zero hits the two-sided 95%
Clopper–Pearson upper limit is $1 - 0.025^{1/N} \approx 3.69/N$.

Decomposition. With $a = P(L = 1)$, $q = P(L = 0, U = 1)$ and $\rho = P(\uparrow g = 1 \mid L = 0, U = 1)$,
$\theta = a + q\rho$ and $\hat\theta = (N_L + N_H)/N$, where $N_L$ counts lanes with $L = 1$ and $N_H$ candidate lanes
whose check succeeds. The product form of #174, $\theta = \pi\rho'$ with $\pi = P(U = 1)$ and
$\rho' = P(\uparrow g \mid U = 1)$, gives the same number ($\hat\pi\hat\rho' = N_U/N \cdot (N_L + N_H)/N_U$). Its delta
method variance is $\rho'^2\pi(1 - \pi)/N + \pi^2\rho'(1 - \rho')/(N\pi) = \theta(1 - \theta)/N$ (the factors are
asymptotically uncorrelated because $E[\hat\rho' \mid N_U] = \rho'$): the product form is useful for reporting, not
for a different interval.

Per-run bounds: $\hat\theta_L = N_L/N$ and $\hat\theta_U = N_U/N$ are unbiased for $P(L)$ and $P(U)$ and satisfy
$\hat\theta_L \le \hat\theta \le \hat\theta_U$ on every run, since the inequalities hold lane by lane.

### 5.3 Subsampled checks

When checks dominate cost, check each candidate lane independently with probability $r \in (0, 1]$ and weight it by
$1/r$. With selection bits $S_k \sim$ Bernoulli($r$) independent of $X$:
$\tilde H_k = L_k + (1 - L_k)U_k S_k \uparrow g(X_k)/r$.

- Unbiased: $E[\tilde H_k \mid X_k] = \uparrow g(X_k)$.
- Variance: $\operatorname{Var}\tilde H_k = \theta(1 - \theta) + \delta(1/r - 1)$, with $\delta = P(L = 0, \uparrow g = 1)$.
  (Derivation: $L$ and the candidate term are disjoint indicators, so $E\tilde H^2 = a + \delta/r$.)
- Interval: $\hat\theta \pm z\,s/\sqrt N$ with $s^2$ the sample variance of $\tilde H$. With few candidate hits the
  normal approximation is poor; use $r = 1$ there.
- Efficiency. Let $c_1$ be the per-lane cost of the bit-parallel stage and $c_2$ the mean cost of one check. The work
  per lane is $c_1 + rqc_2$ and the work-normalised variance is $(c_1 + rqc_2)(V_0 + \delta/r)$ with
  $V_0 = \theta(1 - \theta) - \delta = a(1 - a - 2\delta) - \delta^2$. If $V_0 \le 0$, for example when $a = 0$ and every
  hit comes from a check (the prototype's setting: $L = g$, which is identically 0 on the private model's literal
  target), $r = 1$ is optimal: subsampling is dominated by running fewer lanes. If $V_0 > 0$, the optimum is $r^\star = \min(1, \sqrt{c_1\delta/(qc_2V_0)})$ (§8.4: 1.5× less work at
  $c_2 = 100c_1$).
- Alternative (two-phase): per pass, check a simple random sample of $m$ of the $N_c$ candidates with weight $N_c/m$.
  Conditionally on the lanes this is unbiased, and its extra variance is about $q\rho(1 - \rho)(1 - r)/(rN)$ instead
  of $q\rho(1 - r)/(rN)$: smaller by the factor $1 - \rho$, which matters only when most candidates satisfy the hull.

### 5.4 Cost model

The bit-parallel stage evaluates the cone for $L$ and $U$. $L$ shares every node that does not depend on a
negative-unate event; $U$ shares every node whose cone has no negation. Both add at most one copy of the respective
region, and $g$ itself need not be evaluated. Sampling cost follows edges (#177).

Expected cost per lane is $c_1 + qc_2$ with $q = P(L = 0, U = 1) \le P(g_{coh}) - P(g)$. The prototype, using $g$ and
$g_{coh}$, had $q \approx 3.6\%$. With $L$, $q$ is model-dependent (0 in §8.2).

Check cost: after the fixing of §6.2 the free variables of a lane are a subset of $T(x) \cap B$, with $B$ the binate
events.
Enumeration needs at most $2^{|T(x) \cap B|}$ residual evaluations (64 per word), and by independence
$E[2^{|T(X) \cap B|}] = \prod_{v \in B}(1 + p_v)$ (since $E[2^{X_v}] = 1 + p_v$). Each binate probability-1 event
doubles this bound; that is the failure mode to watch.

### 5.5 Importance sampling

With a product proposal $Q$ (event probabilities $\tilde p_v$) and weight $w(x) = \prod_v P(X_v = x_v)/Q(X_v = x_v)$, the
estimator $N^{-1}\sum_k w(X_k)H_k$, $X_k \sim Q$, is unbiased for $\theta$ provided $\tilde p_v \in (0, 1)$ whenever
$p_v \in (0, 1)$ and $\tilde p_v = p_v$ when $p_v \in \lbrace 0, 1 \rbrace$. $\uparrow g$ is a deterministic function of the
lane, so the check, the bounds and the weights are unchanged: the weight multiplies whatever $H_k$ (or $\tilde H_k$) the
lane produced (#6). The variance is $V_Q/N$, with $V_Q$ the variance of $wH$ under $Q$, estimated by the sample variance; use the
unnormalised form. The candidate fraction under $Q$, $Q(L = 0, U = 1)$, is typically larger than under $P$ when $Q$
raises initiator probabilities, so checks per lane increase.

### 5.6 Determinism

- Lanes are reproducible from (seed, sample index) with counter-based streams per pcfb SPEC §4.
- $H_k$ is a deterministic function of the lane and counts are integers, so $\hat\theta$ is identical for any thread
  count, check order or device that enumerates the same sample indices.
- Subsampling selection is keyed by (seed, global sample index) in a key domain disjoint from the event streams.
- Budgets that choose between check algorithms are counted in deterministic units (BDD nodes, solver conflicts), never
  wall-clock time; they change cost, not answers (§6.4).

## 6. The per-lane check

### 6.1 Problem and complexity

Given a circuit, a target and a world $x$ with true set $T$, decide $\uparrow g(x)$: is $g$ satisfiable with every event
outside $T$ fixed to 0 and every event in $T$ free?

**Theorem 8.** The check is NP-complete. It stays NP-hard when $g = F \wedge \neg D$ with $F$, $D$ monotone and $D$ a
disjunction of pairwise conjunctions (pairwise exclusions), already for $x = 1_V$.

*Proof.* A witness $y$ is checkable in linear time. Hardness, by reduction from CNF-SAT [C71]: for a CNF $\varphi$ over
$u_1, \dots, u_m$ introduce events $u_i^+$, $u_i^-$; let $F$ be $\varphi$ with $u_i \mapsto u_i^+$ and
$\neg u_i \mapsto u_i^-$, conjoined with $\bigwedge_i (u_i^+ \vee u_i^-)$, and let $D = \bigvee_i (u_i^+ \wedge u_i^-)$.
$F$ and $D$ are monotone. A $y$ with $F(y) = 1$ and $D(y) = 0$ makes exactly one of $u_i^+, u_i^-$ true for each $i$,
which encodes an assignment satisfying $\varphi$, and every satisfying assignment gives such a $y$. ∎

**Theorem 9.** Unless P = NP, no polynomial-time algorithm maps a circuit for $g$ to a circuit for $\uparrow g$.

*Proof.* Evaluating the output circuit at $1_V$ would decide CIRCUIT-SAT in polynomial time, since
$\uparrow g(1_V) = 1$ iff $g$ is satisfiable. ∎

Theorem 9 is why the hull is a query kind evaluated per sample, not a compile-time rewrite to a Boolean target.

### 6.2 Algorithm

Once per circuit and target: the cone, a topological order, fan-out lists, and the polarity (positive-unate,
negative-unate, binate) of every event from the dual rail of §4.4.

Per candidate lane:

1. Fix every event outside $T$ to 0.
2. Fix every positive-unate event in $T$ to 1 and every negative-unate event in $T$ to 0. *Lemma:* if some $y \le x$
   satisfies $g$, then raising a positive-unate coordinate to $x_v = 1$ or lowering a negative-unate coordinate to 0
   keeps $g(y) = 1$ and $y \le x$. (This is the pure-literal rule lifted to circuits.)
3. Propagate constants: AND with a 0 input is 0, OR with a 1 input is 1, cardinality bounds shift by the number of
   inputs fixed to 1 and the arity drops by the number fixed, complemented edges and XOR parity fold.
4. If the target is constant, return it. Otherwise optionally recompute polarity on the residual (events can become
   unate after propagation) and repeat 2–3 to a fixpoint.
5. Decide satisfiability of the residual over its free events (a subset of $T \cap B$): enumeration when there are at
   most $k_{enum}$ free events (16–20; 64 assignments per word), else a BDD under a node budget, else complete search
   (DPLL/CDCL on a Tseitin encoding of the residual), which always terminates.

Every branch returns $\uparrow g(x)$ exactly. Lanes are independent and run in parallel (#178).

### 6.3 Why residuals are small

- $E|T \cap \text{cone}| = \sum_v p_v$: apart from high-probability and probability-1 events, few events are true.
- After step 1 every OR whose inputs are all false collapses, and with it most of a fault-tree cone.
- After step 2 only binate events stay free. In delete-term models these are events that occur both in $F$'s cone and
  in some $S$ or $DEL$. Pairwise exclusions among high-probability events are of this kind; flags that occur only under
  negation are not.
- The expected enumeration work is bounded by $\prod_{v \in B}(1 + p_v)$ per lane (§5.4).

When many shared high-probability events are binate, residuals grow and the BDD and complete-search tiers do the work;
Theorem 8 says no algorithm avoids this in the worst case.

### 6.4 Budgets and fallback

Budgets (BDD nodes, solver conflicts) select the next algorithm; the last tier is complete and has no budget. An
optional hard cap per lane (off by default) marks a lane *unresolved*. With $u$ unresolved lanes the run is flagged
incomplete and reports $[N_{hit}/N, (N_{hit} + u)/N]$ widened by the interval of §5.2; unresolved lanes are never
counted as 0 or 1.

### 6.5 Exact evaluation for small models

For up to about 24 events: tabulate $g$ over all worlds, then apply the subset-OR (zeta) transform, for each event $v$
and each world $x$ with $x_v = 1$ setting $H[x] \leftarrow H[x] \vee H[x \oplus 1_{\lbrace v \rbrace}]$ ($n2^n$
operations); $P(\uparrow g) = \sum_x H[x]P(X = x)$. Probability-0 events may be substituted by 0 (Corollary 5.1);
probability-1 events must stay variables. pcfb's test-only `exact_hull_probability` implements this; the reference
tools of #179 can use the same method.

## 7. Interfaces

### 7.1 `ir::pra` quantification config (#181)

- Per target: `negation: boolean | monotone_hull`, default `boolean`. Allowing both on one target is recommended: the
  writer then emits two queries on one tensor (for #185, the canonical hull target and the degeneracy check in one
  file).
- Overrides `element := TRUE | FALSE` produce constants under both semantics; a basic event with literal $p = 1$ stays
  an event. The lowering is function-preserving (#181) and must keep this distinction: house events and overrides
  become constants or `house_event` nodes, basic events with $p \in \lbrace 0, 1 \rbrace$ stay `basic_event` nodes.
- The choice is recorded per pc target on the `pra_pc_lowering` artifact, next to the target bindings.

### 7.2 `ir::pc`: unchanged

Targets do not encode result kinds (`docs/IR-PC-REQUIREMENTS.md` §2). The kind travels from the config to the lowering
artifact, to the writer options (a target → kinds map, like the symbol maps of #176), to `.pcfb` `Query.kind`. A
`.pcfb` → `ir::pc` reader returns kinds in a side table keyed by target; a consumer that cannot carry it must diagnose
non-`PROBABILITY` queries, not drop them.

### 7.3 `.pcfb`: `QueryKind.MONOTONE_HULL_PROBABILITY` (pcfb 1.2)

- **Schema.** `MONOTONE_HULL_PROBABILITY = 1` appended to `QueryKind`; wire-compatible; `flatc --conform` against the
  frozen v1 and v1.1 schemas.
- **Semantics.** §3 over the `SAMPLE` outputs. `BITWISE_ZEROS` / `BITWISE_ONES` are constants; `always_true` samples
  are droppable events.
- **Versioning.** A file using the kind must set `schema_version` minor ≥ 2 (`0x00010002`). The builder writes the
  lowest minor that covers the kinds used, so files without hull queries stay `0x00010001` and byte-identical (fuzzybits
  tests assert `kSchemaVersionV11`, and pcfb fixtures compare committed bytes). Readers accept any minor of major 1.
  `pcfb.opset` stays `core-1.1`: there are no new operators, and a new opset name would make v1.1 validators reject
  files without hull queries as well.
- **Validation** (pcfb SPEC §8 rule 7): every kind is known to the validator, the minor is at least the kind's minor,
  and a hull query is valid on any target.
- **Runtime planning.** A runtime checks the kinds of all queries before sampling and rejects the file if it does not
  implement one. It never answers one kind with another.
- **Compatibility hazard.** The current fuzzybits loader (`src/runtime/pdag_from_pcfb.cc`) ignores `Query.kind` and
  would answer a hull query as a Boolean probability. #183 must add the kind check in the same change as the pcfb pin
  bump, and no fuzzybits writer may emit the kind before that.

### 7.4 Bounds: derived by the runtime

Recommendation: the runtime derives $L$ and $U$ from the target's cone at planning time with the rules of §4.4; the
writer emits no auxiliary query and no query-to-query link. Reasons:

1. **Correctness without trust.** The estimator is unbiased only if $L \le \uparrow g \le U$. Derived by fixed rules,
   the inequalities hold by construction (Theorem 4). A writer-supplied bound $h$ cannot be checked cheaply: deciding
   $\uparrow g \le h$ is coNP-complete (its complement, $\exists x, y : y \le x,\ g(y),\ \neg h(x)$, is in NP and
   contains SAT via $h = 0$). A wrong bound would bias the estimate silently.
2. **Runtime neutrality** (pcfb SPEC §1, goal 3). Bounds belong to one estimator. Exact runtimes (enumeration,
   knowledge compilation) do not use them; the file states what to compute.
3. **Size.** An auxiliary cone duplicates the negation-dependent part of the target cone in the file and in device
   memory; the runtime shares nodes and evaluates $L$ instead of $g$.
4. **No schema surface.** No link field, no validation rule tying two queries together, nothing for transforms to keep
   consistent.
5. **Tightness can improve without a format change.** Writers that want tighter bounds improve the representation
   (for example negation-normal form), which the runtime exploits automatically.

Cost of the recommendation: every hull-capable runtime implements the derivation (one linear pass for polarity plus a
second rail for the negation-dependent nodes), and the bound depends on the representation, which affects cost only.
An optional writer hint would be worth adding only if a measurement shows $L$ and $U$ loose on real models and a writer
can produce provably tighter bounds (§9).

### 7.5 Transform compatibility

Passes declare a semantic class (#177):

- `function_preserving`: $g' = g$ on every world. Preserves every query kind defined from the function, including the
  hull ($\uparrow$ depends only on $g$). It may change $L$, $U$ and $g_{coh}$, hence cost.
- `distribution_preserving`: the target's value is unchanged under the file's independent distribution, for a stated
  set of kinds.

Recommendation: a `distribution_preserving` report lists the kinds it preserves; the planner applies a pass to a cone
only if the pass is function-preserving or every query kind in that cone is in the list. For R1, with targets of both
kinds sharing a cone, apply a pass only if it is acceptable for every kind present.

| Pass | Class | `PROBABILITY` | `MONOTONE_HULL_PROBABILITY` | Reason |
|---|---|---|---|---|
| `propagate_constants`, `eliminate_single_input`, `coalesce` (#177) | function | yes | yes | $g$ unchanged |
| negation normalisation | function | yes | yes | $g$ unchanged; $g_{coh}$ becomes $U$ |
| `fold_zero_probability` | distribution | yes | yes | Corollary 5.1 |
| `fold_unit_probability` | distribution | yes | **no** | Corollary 5.1; §8.2 |
| eliminate a $p = 1$ event by $g \mapsto \exists_v g$ | distribution | no | yes | Theorem 5(b); doubles the cone, not proposed as a pass |
| collapse module $M$ with probability $P(M)$ | distribution | yes | only if $P(\uparrow M) = P(M)$ | Proposition 6.1; §8.7 |
| collapse module $M$ with probability $P(\uparrow M)$ | distribution | only if equal | yes | Proposition 6.1 |

### 7.6 Result reporting

For each hull query a runtime reports:

- the kind (`monotone_hull_probability`), target name, trials $N$, seed;
- $\hat\theta$ and its interval (method and level; Wilson for $r = 1$ without importance sampling, otherwise a CLT
  interval from the sample variance);
- the per-run bounds $\hat\theta_L \le \hat\theta \le \hat\theta_U$, each with an interval, and the bound rules used
  (for example `L=unate`, `U=nnf`);
- the candidate fraction $\hat q$, the number of checked lanes, $r$, and the hit fraction among checked lanes;
- residual statistics: free-event count (median, 95th percentile, maximum), algorithm mix (enumeration, BDD, complete
  search), check time;
- unresolved lanes $u$ (0 for a point estimate, §6.4);
- with importance sampling, the effective sample size.

Projection to PRA entities is as for Boolean results (#181), with the kind in the record.

## 8. Worked examples

All values are exact (rational arithmetic) and were cross-checked by exhaustive enumeration, as were Theorems 1–6 and
8, Proposition 6.1 and the variance formula of §5.3 on random small circuits.

### 8.1 $F = A \vee BC$, $D = AB$, $p_A = 1$, $p_B = 0.1$, $p_C = 0.2$

- $\operatorname{mcs}(F) = \lbrace A \rbrace, \lbrace B, C \rbrace$; $\operatorname{mcs}(D) = \lbrace A, B \rbrace$;
  neither cut set of $F$ contains $\lbrace A, B \rbrace$, so $\uparrow g = A \vee BC$.
- Boolean: $A = 1$ almost surely, so $g = \neg B$ and $P(g) = 0.9$.
- Hull: $P(A \vee BC) = 1$. $U = g_{coh} = F$, $P = 1$.
- Cut-set quantification of $\operatorname{mcs}(\uparrow g)$: REA $= 1 + 0.02 = 1.02 > 1$; MCUB $= 1$ (exact).
- In the world $A = B = C = 1$, $g = 0$ because $D$ holds, but $\uparrow g = 1$: dropping $A$ leaves $\lbrace B, C \rbrace$,
  dropping $B$ leaves $\lbrace A \rbrace$.

### 8.2 Two-flag delete term: $g = (A \vee B) \wedge \neg(XY)$, $p_A = 0.01$, $p_B = 0.02$, $p_X = p_Y = 1$

- Boolean: $XY = 1$ almost surely, so $P(g) = 0$ exactly.
- Hull: $\operatorname{mcs}(F) = \lbrace A \rbrace, \lbrace B \rbrace$, $\operatorname{mcs}(D) = \lbrace X, Y \rbrace$,
  nothing is deleted: $\uparrow g = A \vee B$, $P = 1 - 0.99 \cdot 0.98 = 0.0298$. $U = g_{coh} = A \vee B$.
- `fold_unit_probability` (setting $X$ and $Y$ to TRUE) gives $g' = 0$ and hull 0: wrong.
- Estimator: with $(g, g_{coh})$ every lane with $A \vee B$ is a candidate ($q = 0.0298$). $X$ and $Y$ are
  negative-unate, so $L = (A \vee B) \wedge \neg(0 \wedge 0) = \uparrow g$ and $q = 0$: no checks.
- The pcfb conformance fixture `delete_term.pcfb` is this circuit with $p_A = 0.25$, $p_B = 0.5$: Boolean 0, hull
  0.625, both exact in binary64.

### 8.3 Exclusion between high-probability events: $F = AC \vee BE$, $DEL = AB$, $p_A = p_B = 0.9$, $p_C = p_E = 0.01$

- No cut set of $F$ contains $\lbrace A, B \rbrace$: $\uparrow g = F = g_{coh}$,
  $P(\uparrow g) = 0.009 + 0.009 - 0.000081 = 0.017919$.
- Boolean: $P(g) = P(AC\neg B) + P(BE\neg A) = 0.0009 + 0.0009 = 0.0018$, ten times smaller. The exclusion holds in
  81% of worlds although it deletes no cut set: the private-model effect in miniature.
- $A$ and $B$ are binate, so $L = g$; $q = P(F \wedge AB) = 0.016119$ and every candidate satisfies the hull
  ($\rho = 1$).

### 8.4 A deletion that matters: $F = AB \vee AC \vee E$, $DEL = AB$, $p_A = p_B = 0.5$, $p_C = 0.1$, $p_E = 0.01$

- $\operatorname{mcs}(F) = \lbrace A, B \rbrace, \lbrace A, C \rbrace, \lbrace E \rbrace$; $\lbrace A, B \rbrace$ is
  deleted; $\uparrow g = AC \vee E$.
- $P(g) = P(F) - P(AB) = 0.28225 - 0.25 = 0.03225 < P(\uparrow g) = 1 - 0.95 \cdot 0.99 = 0.0595 < P(g_{coh}) = P(F) = 0.28225$.
- Cut-set quantification: REA $= 0.06$, MCUB $= 0.0595$ (exact, disjoint supports). Without deletion (negations
  dropped): REA $= 0.31$, MCUB $= 0.294625 \ge 0.28225$.
- Estimator: candidates are the worlds with $AB$, $q = 0.25$; $\rho = P(C \vee E) = 0.109$; $\delta = 0.02725$;
  $\theta = a + \delta = 0.03225 + 0.02725$.
- Subsampling: $V_0 = 0.0595 \cdot 0.9405 - 0.02725 = 0.02870975 > 0$, so
  $r^\star = \sqrt{(c_1/c_2) \cdot 0.02725/(0.25 \cdot 0.02870975)} \approx 1.948\sqrt{c_1/c_2}$. At $c_2 = 100c_1$,
  $r^\star \approx 0.195$ and the work-normalised variance is $0.990c_1$ against $1.455c_1$ at $r = 1$.

### 8.5 Two sequences of one event tree: $s_1 = IA\neg B$, $s_2 = IAB$, $p_I = 0.01$, $p_A = 0.1$, $p_B = 0.2$

- Boolean: $0.0008 + 0.0002 = 0.001 = P(IA)$; the sequences are disjoint.
- Hull: $\uparrow s_1 = IA$ ($0.001$; the success branch deletes nothing because no cut set of $IA$ contains $B$),
  $\uparrow s_2 = s_2$ ($0.0002$). The class $\uparrow(s_1 \vee s_2) = IA$ has $0.001$, not the sum $0.0012$.

### 8.6 Probability-0 event: $g = (A \vee Z) \wedge \neg(ZB)$, $p_A = 0.1$, $p_Z = 0$, $p_B = 0.5$

$P(\uparrow g) = 0.1$; after folding $Z :=$ FALSE, $\uparrow(A)$ has $0.1$ as well (Corollary 5.1).

### 8.7 Non-monotone module: $M = A\neg B$, $g = M \vee C$, $p_A = 0.5$, $p_B = 0.25$, $p_C = 0.1$

$P(\uparrow g) = P(A \vee C) = 0.55$. Collapsing $M$ with $P(M) = 0.375$ gives $1 - 0.625 \cdot 0.9 = 0.4375$ (wrong);
with $P(\uparrow M) = P(A) = 0.5$ it gives $0.55$ (Proposition 6.1).

## 9. Open questions

1. Both kinds on one target in the config (§7.1): recommended; confirm in #181.
2. The fuzzybits kind check must land with the pin bump in #183 (§7.3).
3. A writer hint for tighter bounds: deferred until $L$ and $U$ are measured on the private model (§7.4).
4. Defaults: $k_{enum}$, BDD and solver budgets, whether to expose subsampling ($r < 1$) and the two-phase variant in
   R1, interval choice for $r < 1$.
5. Whether residual polarity is recomputed per lane (§6.2 step 4): a #178 measurement.
6. Initiator-tagged events in FTP projects (#185): FTREX's rules may differ from plain delete-term semantics; any
   remaining gap between $P(\uparrow g)$ and cut-set results must be explained by those rules, truncation (§4.7), or
   REA/MCUB.
7. Hull variants of later query kinds (importance measures) are out of scope.

## 10. References

- [C71] S. A. Cook, "The complexity of theorem-proving procedures," *Proc. 3rd ACM STOC*, pp. 151–158, 1971.
  doi:10.1145/800157.805047.
- [CP34] C. J. Clopper, E. S. Pearson, "The use of confidence or fiducial limits illustrated in the case of the
  binomial," *Biometrika* 26(4):404–413, 1934. doi:10.1093/biomet/26.4.404.
- [DR97] Y. Dutuit, A. Rauzy, "Exact and truncated computations of prime implicants of coherent and non-coherent fault
  trees within Aralia," *Reliability Engineering & System Safety* 58:127–144, 1997. doi:10.1016/S0951-8320(97)00034-3.
- [EP63] J. D. Esary, F. Proschan, "Coherent structures of non-identical components," *Technometrics* 5:191–209, 1963.
  doi:10.1080/00401706.1963.10490075.
- [EPW67] J. D. Esary, F. Proschan, D. W. Walkup, "Association of random variables, with applications," *Ann. Math.
  Statist.* 38(5):1466–1474, 1967. doi:10.1214/aoms/1177698701.
- [KSC22] J. Knudsen, C. Smith, M. Calley, "Issues and approaches regarding success terms for probabilistic risk
  assessment models," PSAM 16, Honolulu, 2022. https://www.iapsam.org/PSAM16/papers/JA278-PSAM16.pdf
- [MEF] Open-PSA Model Exchange Format 2.0, §6.2 "Delete Terms, Recovery Rules, and Exchange Events."
  https://open-psa.github.io/mef/mef/meta_logical_layer.html
- [R01] A. Rauzy, "Mathematical foundations of minimal cutsets," *IEEE Transactions on Reliability* 50(4):389–396, 2001.
  doi:10.1109/24.983400. Section and theorem numbers above refer to the author's preprint
  (https://antoira.folk.ntnu.no/Publications/pdf/Rauzy2001-MathematicaFoundationOfMinimalCutsets.pdf).
- [V79] L. G. Valiant, "The complexity of enumeration and reliability problems," *SIAM Journal on Computing* 8:410–421,
  1979.
- [W27] E. B. Wilson, "Probable inference, the law of succession, and statistical inference," *Journal of the American
  Statistical Association* 22:209–212, 1927.
