# RFC: monotone-hull (delete-term) probability queries

Status: proposed  
Issue: [#174](https://github.com/fuzzybit-org/fuzzybits/issues/174) · amendment:
[#190](https://github.com/fuzzybit-org/fuzzybits/issues/190) · tracking:
[#185](https://github.com/fuzzybit-org/fuzzybits/issues/185)  
Related: [#177](https://github.com/fuzzybit-org/fuzzybits/issues/177) (pass semantic classes),
[#178](https://github.com/fuzzybit-org/fuzzybits/issues/178) (per-assignment check),
[#179](https://github.com/fuzzybit-org/fuzzybits/issues/179) (reference estimators, fixtures),
[#181](https://github.com/fuzzybit-org/fuzzybits/issues/181) (quantification config),
[#183](https://github.com/fuzzybit-org/fuzzybits/issues/183) (runtime),
[#6](https://github.com/fuzzybit-org/fuzzybits/issues/6) (importance sampling)  
Format: `.pcfb` `QueryKind.MONOTONE_HULL_PROBABILITY`, pcfb 1.2
([`SPEC-v1.2.md`](https://github.com/fuzzybit-org/pcfb/blob/main/docs/SPEC-v1.2.md),
[pcfb#10](https://github.com/fuzzybit-org/pcfb/pull/10)); literal modes need an amendment to 1.2 (§7.3)

## 0. Summary

PRA models built for cut-set tools write event-tree success branches and delete terms with negation, but they mean
cut-set deletion, not Boolean complement. This RFC defines hull queries on circuits with an explicit set of *literal
atoms*: complemented basic-event operands that the source marks as atomic literals. Under the **atom-split hull** $h$,
the default interpretation of marked literals, each marked complement $\bar B$ is an atom of its own, valued
$\neg x_B$ and droppable independently of $B$, and $h$ is the plain monotone hull over all atoms, evaluated on the
coupled sample; every unmarked complement is an ordinary negation, i.e. a delete term. Without marked literals $h$ is
the *plain hull* $\uparrow g$, the least monotone function above $g$. The FTP reader marks every `-EVENT` reference
(never `-GATE`), because cut-set tools keep `-EVENT` in cut sets as an event of its own and treat `-GATE` as a delete
term; other sources mark nothing unless their format defines such literals (#199). The RFC proves that the minimal cut
sets of $h$ are the cut sets a delete-term tool keeps (Theorem 10), adds the tools' at-most-one-initiator rule as a
delete term, gives an unbiased Monte Carlo estimator of $P(h)$ that runs an exact check only on samples where two cheap
bounds disagree, and fixes the interfaces. Two other interpretations of marked literals are explicit options: Rauzy's
widening $\omega_L g$ [R01], which fixes the marked events, and `negation`, which reads them as ordinary negations
(the plain hull).

Decisions proposed here:

1. **A query kind, not a rewrite.** `MONOTONE_HULL_PROBABILITY` is a second probability query kind (pcfb 1.2). No
   polynomial-time compiler can turn a circuit for $g$ into a circuit for its hull unless P = NP (§6.1).
2. **Bounds are derived by the runtime**, not emitted by the writer (§7.4): a lower bound $\underline g$ (unate events
   fixed to their best value) and an upper bound $\overline g$ (negation-normal relaxation), both correct by
   construction.
3. **pcfb 1.2**: one enum value appended; `schema_version` `0x00010002` only when the kind is used; opset unchanged;
   runtimes reject kinds they do not implement before sampling (§7.3). The literal mode and the literal samples are an
   amendment to 1.2; without them the query is the plain hull of pcfb 1.2 as merged.
4. **Transforms** state which query kinds they preserve. Hull targets accept passes that preserve the function over
   atoms and folding of variables that are 0 almost surely; a variable that is 1 almost surely may be folded only when
   that is shown to be safe (§7.5).
5. **`ir::pc` is unchanged**; the kind, the literal mode and the initiator rule are chosen per target in the
   `ir::pra` quantification config (§7.1, §7.2).
6. **Literal atoms come from the source** (§2, §3, #199). `ir::pra` stays source-neutral: the FTP reader marks
   complemented basic-event operands with the operand flag `literal` and records its conventions as inert model
   attributes; the lowering does not infer literals. Per hull target the interpretation of marked literals is
   `atom_split` (default), `omega_l` or `negation`; the lowering warns (`complement-under-negation`) where a marked
   event also occurs under a negated gate, the only place `atom_split` and `omega_l` can differ (Theorem 10(c)).
7. **Initiator rule** (§4.8): `max_initiators_per_cutset = m` is lowered as the delete term
   $\neg\,\mathrm{atleast}(m+1;\ \text{initiating events})$; under the hull it deletes exactly the minimal cut sets with
   more than $m$ initiators (Corollary 3.4, Theorem 10(f)).

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
- The model has no direct complement of a basic event (all its complements are on gates), so for it the query of §3
  is the plain hull.

**Vendor semantics.** The vendor tools of #185, run on the public conformance fixtures of #179 and on small probe
models, treat a complemented gate by delete-term processing, including a complemented single-input gate; keep a
directly complemented basic event `-B` as a cut-set member of probability $1 - p_B$ and remove cut sets that contain
both $B$ and `-B` (a mutually exclusive event check); list no consensus cut sets; keep probability-1 events as ordinary
events in cut sets; and delete cut sets with two or more initiating events (`I`-tagged in `IMPORT`; probability-1
initiator flags count). Their quantification configuration declares the initiator rule per engine (on for the engine
used here, off for others). The atom-split hull of §3 reproduces the tools' cut sets on every fixture and every probe
(§8.8–§8.11, Theorem 10). The FTP reader therefore marks every `-EVENT` as a literal. Hull queries and the initiator
rule are requested, never implied by the reader (#244, #262).

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
is the set of prime implicants; $g$ is the disjunction of $\operatorname{PI}(g)$. A prime implicant mentions only
variables $g$ depends on. For a family $\mathcal{S}$ of sets, $\operatorname{Min}\mathcal{S}$ is its set of
inclusion-minimal members. $\lambda(x) = \lbrace x_v : x_v = 1 \rbrace \cup \lbrace \neg x_v : x_v = 0 \rbrace$ is the
set of literals true at $x$ (the minterm of $x$).

**Significant literals** [R01, §IV.C]. A set $L$ of literals splits $V$ into $P_L$ (only $x_v \in L$), $B_L$ (both
$x_v, \neg x_v \in L$), $N_L$ (only $\neg x_v \in L$) and $O_L$ (neither). $L^+ = \lbrace x_v : v \in V \rbrace$ and
$L^\pm$ is the set of all literals. An *$L$-term* is a term $t \subseteq L$. The preorder

$$y \sqsubseteq_L x \iff \lambda(y) \cap L \subseteq \lambda(x)$$

says that every significant literal true at $y$ is true at $x$. Per event it reads $y_v \in A_v^L(x_v)$ with

| class | $A_v^L(x_v)$ | in words |
|---|---|---|
| $P_L$ | $\lbrace 0, x_v \rbrace$ ($y_v \le x_v$) | droppable |
| $B_L$ | $\lbrace x_v \rbrace$ ($y_v = x_v$) | fixed |
| $N_L$ | $\lbrace x_v, 1 \rbrace$ ($y_v \ge x_v$) | raisable |
| $O_L$ | $\lbrace 0, 1 \rbrace$ | free |

so $\sqsubseteq_L$ is a product of per-event relations; it is reflexive and transitive, antisymmetric iff
$O_L = \emptyset$, and $\sqsubseteq_{L^+}$ is $\le$. If $L \subseteq L'$ then $y \sqsubseteq_{L'} x$ implies
$y \sqsubseteq_L x$. A function is *$L$-monotone* if $y \sqsubseteq_L x$ implies $g(y) \le g(x)$: monotone in $P_L$,
antitone in $N_L$, independent of $O_L$, unconstrained in $B_L$. Monotone functions of $L$-literals are $L$-monotone.

Following [R01, §IV.C]: an $L$-term $t$ is an *$L$-cut set* of $g$ if every world $x$ satisfying $t$ has some
$y \sqsubseteq_L x$ with $g(y) = 1$; $\operatorname{MC}_L(g)$ is the family of inclusion-minimal $L$-cut sets (the
*minimal cut sets*), and $\operatorname{PI}_L(g) = \operatorname{Min}\lbrace t \cap L : t \in \operatorname{PI}(g) \rbrace$
is the family of minimal $L$-parts of prime implicants. A negative literal in a cut set is written `-NAME` in tool
output; the probability of an $L$-term $C$ is $P(C) = \prod_{x_v \in C} p_v \prod_{\neg x_v \in C} (1 - p_v)$.

Circuits are DAGs over $V$ and constants with the `ir::pc` / `.pcfb` connectives: AND, OR, XOR, NOT, NAND, NOR, XNOR,
cardinality (`ATLEAST`, `ATMOST`, `EXACTLY`, or `cardinality(low, high)`), and complemented edges (`edge_ref.complement`,
pcfb `inverted_inputs`). A target computes a function $g$; its *cone* is the set of nodes it depends on. A *direct
complement* is a complemented edge whose child is an event (FTP `-NAME` with `NAME` an imported event); a complemented
edge into a gate, including a single-input gate, is a *gate complement*.

**Marked literals and atoms.** A *marked literal* is a direct complement that the source marks as an atomic literal of
the event (`ir::pra` operand flag `literal`, #199); the flag does not change the Boolean meaning, and is allowed only on
complemented operands that reference basic events. The FTP reader marks every direct complement (`-EVENT`; reader option
`io.ftp.complement_events = atom | negation`, default `atom`) and no gate complement; other readers mark nothing unless
their format defines such literals. For a target with circuit $C$, let $\Delta$ be the events with a marked literal in
its cone; unmarked complements are ordinary negations.
The *atoms* are $\mathcal{A} = V \cup \lbrace \bar v : v \in \Delta \rbrace$. The *split circuit* $\tilde C$ replaces
every marked literal of an event $v \in \Delta$ by a positive edge into the atom $\bar v$; every other complement stays. It
computes $\tilde g : \lbrace 0, 1 \rbrace^{\mathcal{A}} \to \lbrace 0, 1 \rbrace$. The *coupled valuation* of a world $x$
is $\kappa(x) \in \lbrace 0, 1 \rbrace^{\mathcal{A}}$ with $\kappa(x)_v = x_v$ and $\kappa(x)_{\bar v} = 1 - x_v$, so
$g = \tilde g \circ \kappa$. The atom $\bar v$ has probability $q_{\bar v} = 1 - p_v$, and $q_v = p_v$; the atoms of
$\kappa(X)$ are not independent, since $\kappa(X)_{\bar v} = 1 - \kappa(X)_v$. Tool output writes $\bar v$ as `-NAME`.

Restriction and quantification: $g_{v \leftarrow c}(x) = g(x[v \leftarrow c])$ and $\exists_v g = g_{v \leftarrow 0} \vee g_{v \leftarrow 1}$.

## 3. Definition

The query is a monotone hull of a circuit with an explicit set $\Delta$ of literal atoms (§2). The *literal mode*
says how the marked literals are read; it does not affect unmarked complements, which are always ordinary negations.
With no marked literal, $\Delta = \emptyset$ and every mode gives the plain hull $\uparrow g$; a model from a source
without such literals gets the plain hull.

**Atom-split hull (`atom_split`, the default interpretation of marked literals).**

$$h(x) = \uparrow\tilde g(\kappa(x)) = \bigvee_{(y, w) \le \kappa(x)} \tilde g(y, w):$$

the plain monotone hull of the split circuit over all atoms, evaluated at the coupled valuation. $h(x) = 1$ iff some
subset of the atoms that are true at $\kappa(x)$ (the true events, and $\bar v$ for every false $v \in \Delta$), with
every other atom false, satisfies $\tilde g$. A marked literal `-B` is thus an atom of its own, which a witness may
drop independently of $B$; unmarked complements, including every gate complement, are delete terms. The minimal cut sets are taken over atoms:
$K_h = \lbrace C \in \operatorname{mcs}(\uparrow\tilde g) : \lbrace v, \bar v \rbrace \not\subseteq C \text{ for every } v \rbrace$
(a set with both atoms of an event has probability 0). The query value is $P(h(X))$.

**Rauzy's widening (option `omega_l`).**

$$\omega_L g(x) = \bigvee_{y \sqsubseteq_L x} g(y).$$

$\omega_L g(x) = 1$ iff $g$ holds at some world that agrees with $x$ on the fixed events and is obtained from $x$ by
switching off droppable events that are true in $x$ (and switching on raisable events, setting free events at will).
Rauzy calls $\omega_L$ the widening of $g$ with respect to $L$: it adds to $g$ every world that is "more significant"
than a world already in $g$ [R01, §IV.D]. The option uses the model's literals
$L_g = L^+ \cup \lbrace \neg x_v : v \in \Delta \rbrace$: the events with marked literals are fixed, every other event
is droppable, $N_{L_g} = O_{L_g} = \emptyset$. Its minimal cut sets are $\operatorname{MC}_{L_g}(g)$.

**Plain hull (option `negation`).** The marked literals are read as ordinary negations:
$\uparrow g = \omega_{L^+} g$, i.e. $\uparrow g(x) = \bigvee_{y \le x} g(y)$: some subset of the events that are true
in $x$, with every other event false, satisfies $g$.

Without marked literals, $\Delta = \emptyset$, $\tilde g = g$, $L_g = L^+$, and all three are $\uparrow g$. The
reference tools of #179, which read only FTP, spell the modes `split` (default), `omega` and `positive`.

Remarks:

- *Why atoms, and why the FTP reader marks them.* For FTP models, the tools keep `-EVENT` in cut sets, drop it like
  any other member, remove cut sets with both $B$ and `-B`, and treat `-GATE` (also a single-input gate) as a delete
  term: delete-term processing over atoms (Theorem 10(a)). This reproduces every fixture and probe, including those
  where $\omega_{L_g}$ does not (a marked event that also occurs under a negated gate) and the absence of consensus cut
  sets (§4.9). It is an FTP convention: the reader applies it by marking, and `ir::pra` and the lowering only carry
  the marks.
- *Where the modes differ.* $g \le \omega_{L_g} g \le h$ always, and $h = \omega_{L_g} g$ unless a marked event also
  occurs in the cone of a negated gate (Theorem 10(b, c)); the lowering warns there (§7.1). $h \le \uparrow g$ fails in
  general (Theorem 10(d)).
- *Why the cone.* A target's value does not depend on logic outside its cone. An event complemented elsewhere but used
  only positively here changes nothing (`T3` in §8.8).
- *The literal mode and $\Delta$ are part of the query*, fixed when the target is lowered (§7.1). $h$ depends only on
  $\tilde g$ and the coupling, $\omega_L g$ only on $g$ and $L$: transforms must preserve $\tilde g$ for `atom_split`
  and $g$ for the other modes (§7.5).
- *Between Boolean and plain hull, for the option.* By Theorem 1(d),
  $g = \omega_{L^\pm} g \le \omega_{L_g} g \le \uparrow g$.

## 4. Properties

### 4.1 Least $L$-monotone majorant

**Theorem 1.** For every $L$: (a) $\omega_L g$ is $L$-monotone and $g \le \omega_L g$. (b) If $h$ is $L$-monotone and
$g \le h$, then $\omega_L g \le h$. (c) $\omega_L$ is a closure operator: $g \le g'$ implies
$\omega_L g \le \omega_L g'$; $\omega_L\omega_L g = \omega_L g$; $\omega_L g = g$ iff $g$ is $L$-monotone.
(d) $L \subseteq L'$ implies $\omega_{L'} g \le \omega_L g$.

*Proof.* (a) If $x \sqsubseteq_L x'$, every $y \sqsubseteq_L x$ satisfies $y \sqsubseteq_L x'$ by transitivity, so
$\omega_L g(x) \le \omega_L g(x')$; taking $y = x$ (reflexivity) gives $g \le \omega_L g$. (b) If $\omega_L g(x) = 1$
there is $y \sqsubseteq_L x$ with $g(y) = 1$, so $h(y) = 1$ and, $h$ being $L$-monotone, $h(x) = 1$. (c) Monotonicity of
$\omega_L$ is immediate from the definition. By (a) $\omega_L g$ is an $L$-monotone majorant of itself, so (b) with
$h = \omega_L g$ gives $\omega_L\omega_L g \le \omega_L g$, and (a) gives the reverse. If $g$ is $L$-monotone, (b) with
$h = g$ gives $\omega_L g \le g$; conversely $\omega_L g$ is $L$-monotone. (d) $\sqsubseteq_{L'}$ is contained in
$\sqsubseteq_L$ (§2). ∎

With $L = L^+$: $\uparrow g$ is the least monotone function above $g$, and $\uparrow g = g$ iff $g$ is monotone.

**Lemma 1.1.** Every prime implicant of an $L$-monotone function $h$ is an $L$-term.

*Proof.* Let $t \in \operatorname{PI}(h)$ contain a literal $\ell \notin L$ of variable $v$, and $t' = t \setminus \lbrace \ell \rbrace$.
Let $x$ satisfy $t'$. If $x$ satisfies $\ell$ it satisfies $t$, so $h(x) = 1$. Otherwise the world $x'$ obtained by
flipping $x_v$ satisfies $t$, so $h(x') = 1$; $\lambda(x')$ and $\lambda(x)$ differ only at $v$, where $\lambda(x')$
has $\ell \notin L$, so $\lambda(x') \cap L \subseteq \lambda(x)$, i.e. $x' \sqsubseteq_L x$, and $h(x) = 1$. So $t'$
is an implicant, contradicting primality. ∎

**Lemma 1.2 (fixed events are parameters).** Write $x = (x_B, x_R)$ with $B = B_L$, $R = V \setminus B$, and $L_R$ for
the literals of $L$ over $R$. Then $\omega_L g(x) = \omega_{L_R}\bigl(g(x_B, \cdot)\bigr)(x_R)$. In particular
$\omega_L(g \wedge h) = \omega_L g \wedge h$ if $h$ depends only on fixed events.

*Proof.* $y \sqsubseteq_L x$ iff $y_B = x_B$ and $y_R \sqsubseteq_{L_R} x_R$. For the second claim, $h(y) = h(x)$ for
every $y \sqsubseteq_L x$. ∎

### 4.2 Minimal cut sets

**Theorem 2.** For every $g$ and $L$:

- (a) $\operatorname{MC}_L(g) = \operatorname{PI}(\omega_L g)$ [R01, Theorem 2];
- (b) $\omega_L g = \bigvee_{t \in \operatorname{PI}_L(g)} t = \bigvee_{C \in \operatorname{MC}_L(g)} C$;
- (c) if no event occurs with both polarities among the members of $\operatorname{PI}_L(g)$, in particular if
  $B_L = \emptyset$, then $\operatorname{MC}_L(g) = \operatorname{PI}_L(g)$ [R01, Theorem 1];
- (d) $\operatorname{mcs}(\uparrow g) = \operatorname{Min}\lbrace \operatorname{pos}(t) : t \in \operatorname{PI}(g) \rbrace = \operatorname{Min}\lbrace S : g(1_S) = 1 \rbrace$.

Theorem numbers refer to the author's preprint of [R01]. With $L = L^+$ the minimal cut sets are the minimal positive
parts of the prime implicants, the usual definition for non-coherent functions and the *minimal p-cuts* of [DR97].

*Proof.*

1. For $t \in \operatorname{PI}(g)$ the $L$-term $t \cap L$ implies $\omega_L g$: given $x$ satisfying $t \cap L$, let
   $y$ be $x$ with the literals of $t \setminus L$ imposed. $y$ satisfies $t$, so $g(y) = 1$, and $y$ differs from $x$
   only at variables whose literal in $\lambda(y)$ lies in $t \setminus L$, so $\lambda(y) \cap L \subseteq \lambda(x)$.
2. If $\omega_L g(x) = 1$, take $y \sqsubseteq_L x$ with $g(y) = 1$. The minterm $\lambda(y)$ is an implicant of $g$ and
   contains a prime implicant $t$, and $t \cap L \subseteq \lambda(y) \cap L \subseteq \lambda(x)$, so $x$ satisfies
   $t \cap L$ and hence a member of $\operatorname{PI}_L(g)$. With step 1 this is the first equality of (b).
3. (a) By definition an $L$-term is an $L$-cut set iff it implies $\omega_L g$. Every $L$-term implicant contains a
   prime implicant, which is again an $L$-term; conversely the prime implicants of the $L$-monotone function
   $\omega_L g$ (Theorem 1(a)) are $L$-terms (Lemma 1.1). So the minimal $L$-cut sets are the prime implicants of
   $\omega_L g$, whose disjunction is $\omega_L g$: the second equality of (b).
4. (c) Replace $x_v$ by $\neg x_v$ for every event that occurs only negatively among the members of
   $\operatorname{PI}_L(g)$. In the new variables these members form an antichain $K$ of positive terms and, by (b),
   $\omega_L g = \bigvee K$ is monotone. A positive term $s$ implies $\bigvee K$ iff $s$ contains a member of $K$
   (evaluate at the world where exactly the variables of $s$ are 1), and the prime implicants of a monotone function
   are positive (Lemma 1.1 with $L^+$), so $\operatorname{PI}(\omega_L g) = \operatorname{Min} K = K$. With (a),
   $\operatorname{MC}_L(g) = \operatorname{PI}_L(g)$.
5. (d) For $L = L^+$, (a) and (c) give $\operatorname{mcs}(\uparrow g) = \operatorname{PI}_{L^+}(g)$, which is the first
   expression. $\uparrow g(1_S) = 1$ iff some $S' \subseteq S$ has $g(1_{S'}) = 1$, so the cut sets of $\uparrow g$ are
   the up-closure of $\lbrace S : g(1_S) = 1 \rbrace$, whose minimal members are the second expression. ∎

Example [R01]: $g = ab \vee \neg a c$ has prime implicants $ab$, $\neg a c$, $bc$; the minimal positive parts are $ab$
and $c$, so $\uparrow g = ab \vee c$.

*Remark (Theorem 2(c) needs its hypothesis).* [R01] states Theorem 1 for every $L$. It fails when an event of $B_L$
occurs with both polarities: for $g = AB\neg C \vee A\neg BC$ with $L = L^+ \cup \lbrace \neg x_B \rbrace$,
$\operatorname{PI}(g) = \lbrace AB\neg C, A\neg BC \rbrace$ and $\operatorname{PI}_L(g) = \lbrace AB, A\neg BC \rbrace$,
but $\omega_L g = AB \vee A\neg BC$ has the implicant $AC$ (if $A$ and $C$ hold, either $B$ or $\neg B$ completes a
term), so $\operatorname{MC}_L(g) = \lbrace AB, AC \rbrace$. The disjunctions agree, as (b) requires. The same example
falls under case 1 of [R01, Theorem 3], whose recursion returns $\operatorname{PI}_L(g)$. Theorem 2(a) holds as stated;
this RFC defines the minimal cut sets as $\operatorname{MC}_L(g)$.

### 4.3 Delete terms

**Lemma 3.0.** If $D$ is $L$-monotone, $\omega_L(g \wedge \neg D) = \omega_L(\omega_L g \wedge \neg D)$.

*Proof.* "$\le$" by Theorem 1(a, c). "$\ge$": if $y \sqsubseteq_L x$, $D(y) = 0$, and $z \sqsubseteq_L y$ has $g(z) = 1$,
then $D(z) \le D(y) = 0$ and $z \sqsubseteq_L x$. ∎

**Theorem 3.** Let $F$ and $D$ be $L$-monotone, $D$ independent of the fixed events $B_L$, and $g = F \wedge \neg D$.
Let $K = \lbrace C \in \operatorname{MC}_L(F) : \text{no } C' \in \operatorname{MC}_L(D) \text{ has } C' \subseteq C \rbrace$.
Then $\omega_L g = \bigvee_{C \in K} C$. If no event occurs with both polarities among the members of $K$ (always the
case for $L = L^+$), $\operatorname{MC}_L(g) = K$.

*Proof.* By Theorem 1(c) and Theorem 2(a), $\operatorname{MC}_L(F) = \operatorname{PI}(F)$ and
$\operatorname{MC}_L(D) = \operatorname{PI}(D)$; they are $L$-terms (Lemma 1.1), and those of $D$ have no literal of
$B_L$.

- "$\ge$": let $x$ satisfy $C \in K$. Let $y$ agree with $x$ on $B_L \cup O_L$; on $P_L$ let $y_v = 1$ iff
  $x_v \in C$; on $N_L$ let $y_v = 0$ iff $\neg x_v \in C$. Then $y \sqsubseteq_L x$ and $y$ satisfies $C$, so
  $F(y) = 1$. The literals of $L$ over $P_L \cup N_L$ that are true at $y$ are exactly those in $C$. If $D(y) = 1$, some
  $C' \in \operatorname{PI}(D)$ holds at $y$; its literals lie over $P_L \cup N_L$, so $C' \subseteq C$, contradicting
  $C \in K$. So $g(y) = 1$.
- "$\le$": let $y \sqsubseteq_L x$ with $F(y) = 1$ and $D(y) = 0$. Some $C \in \operatorname{PI}(F)$ holds at $y$, and
  $C \subseteq \lambda(y) \cap L \subseteq \lambda(x)$, so $C$ holds at $x$. If some $C' \in \operatorname{PI}(D)$ had
  $C' \subseteq C$, then $D(y) = 1$. So $C \in K$.

$K$ is a subfamily of the antichain $\operatorname{PI}(F)$; under the polarity hypothesis step 4 of Theorem 2 gives
$\operatorname{PI}(\bigvee K) = K$, and $\operatorname{MC}_L(g) = \operatorname{PI}(\omega_L g) = K$. ∎

**Corollary 3.1 (any target).** For any $g$ and any $L$-monotone $D$ independent of $B_L$,
$\omega_L(g \wedge \neg D) = \bigvee \lbrace C \in \operatorname{MC}_L(g) : \text{no } C' \in \operatorname{MC}_L(D) \text{ has } C' \subseteq C \rbrace$.

*Proof.* Lemma 3.0 and Theorem 3 with $F = \omega_L g$, whose minimal cut sets are $\operatorname{PI}(\omega_L g) = \operatorname{MC}_L(g)$. ∎

**Corollary 3.2 (sequences).** For $SEQ = F \wedge \neg(S_1 \vee \dots \vee S_m \vee DEL)$ with $F$, $S_i$, $DEL$ built
from AND, OR and `ATLEAST` over events and marked literals, and no event with a marked literal in the cones of the
$S_i$ and $DEL$, the hull keeps exactly the minimal cut sets of $F$ that contain no minimal cut set of any success
system $S_i$ and no minimal cut set of $DEL$. For a delete-term group $\lbrace e_1, \dots, e_k \rbrace$ written as
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
overestimate when success probabilities are not close to 1). Theorem 3 needs $F$ and $D$ to be $L$-monotone and $D$ to
leave the fixed events alone; Theorem 2 defines the hull for any $g$.

*Remark (fixed events inside a delete term).* Fixed events stay fixed inside $D$ as well. For $F = A$,
$D = A \wedge \neg B$ and $\neg x_B \in L$, $\omega_L(F \wedge \neg D) = \omega_L(AB) = AB$, while $K = \lbrace A \rbrace$.
In general, by Lemma 1.2, Theorem 3 holds for each assignment $b$ of the fixed events, with $F(b, \cdot)$ and
$D(b, \cdot)$ in place of $F$ and $D$.

The atom-split hull is the plain hull of the split circuit, so Theorem 3 with $L^+$ over atoms gives its cut sets
directly, without the hypothesis on $D$ (Theorem 10(a)); the Remark's example is where it and $\omega_L$ part.

### 4.4 Bounds

Three functions, each computable by one bit-parallel pass over a circuit:

- **Edge relaxation $g_{coh}^L$.** Replace every gate complement by TRUE; a direct complement of $v$ by TRUE if
  $\neg x_v \notin L$ and keep it (the literal $\neg x_v$) otherwise; a positive edge into an event $v$ with
  $x_v \notin L$ by TRUE; every NOT, NAND, NOR, XNOR and `ATMOST` by TRUE, XOR by OR, `EXACTLY k` by `ATLEAST k`, and
  `cardinality(low, high)` by `ATLEAST low`. $g_{coh} = g_{coh}^{L^+}$ is the bound named in #174 ("negated inputs
  replaced by TRUE"); $g_{coh}^{L_g}$ relaxes only gate negations and keeps direct complements.
- **Negation-normal relaxation $\overline g$.** Give every node $v$ a positive rail $v^+$ (computing $v$) and a negative
  rail $v^-$ (computing $\neg v$), built only from AND, OR and `ATLEAST`: for AND, $v^+ = \bigwedge \ell_j$ and
  $v^- = \bigvee \bar\ell_j$, where $\ell_j$ is the child's positive rail (negative rail if the edge is complemented)
  and $\bar\ell_j$ the opposite rail; OR dually; NOT swaps rails; `ATLEAST k` of $m$ has
  $v^+ = [\sum_j \ell_j \ge k]$, $v^- = [\sum_j \bar\ell_j \ge m-k+1]$; NAND, NOR, XNOR, `ATMOST`, `EXACTLY` and
  `cardinality(low, high)` follow by De Morgan; XOR uses the parity chain $P_j = P_{j-1}\bar\ell_j \vee N_{j-1}\ell_j$,
  $N_j = P_{j-1}\ell_j \vee N_{j-1}\bar\ell_j$. Leaves: $x_v^+ = x_v$, $x_v^- = z_v$. Then
  $g(x) = G(x, \neg x)$ for the monotone function $G(x, z)$ computed by the target's positive rail, and
  $\overline g(x) = G(a(x), b(x))$ with $a_v = x_v$ if $x_v \in L$ else 1, and $b_v = \neg x_v$ if $\neg x_v \in L$
  else 1: every literal outside $L$ relaxed to TRUE. For $L^+$ this is $G(x, 1)$. A node whose cone contains no
  negation has $v^+ = v$ under the relaxation, and its negative rail is the constant $\neg v(0)$, so $\overline g$
  costs extra only where negations reach.
- **Unate fixing $\underline g$.** Call $v$ *negative-unate* if leaf $x_v$ does not occur in the target's positive rail
  (only $z_v$ does), *positive-unate* if $z_v$ does not occur, *binate* if both occur. $\underline g(x) = g(\tilde x)$,
  where $\tilde x$ sets negative-unate droppable events to 0, positive-unate raisable events to 1, unate free events to
  their better value, and keeps every other coordinate, in particular every fixed event.

**Theorem 4.** Pointwise, $g \le \underline g \le \omega_L g \le \overline g \le g_{coh}^L$; $\overline g$ and
$g_{coh}^L$ are $L$-monotone. If $L \supseteq L^+$, also $g_{coh}^L \le g_{coh}$.

*Proof.*

- Unateness: $G$ is monotone in $(x, z)$. If $v$ is negative-unate, $G$ does not depend on $x_v$, so
  $g(x) = G(x, \neg x)$ is antitone in $x_v$; symmetrically for positive-unate.
- $g \le \underline g$: each changed coordinate moves in the direction in which $g$ cannot decrease.
- $\underline g \le \omega_L g$: every change stays in $A_v^L(x_v)$, so $\tilde x \sqsubseteq_L x$.
- $\omega_L g \le \overline g$: for $y \sqsubseteq_L x$, per event $y_v \le a_v(x)$ and $\neg y_v \le b_v(x)$ (check
  the four classes), so $g(y) = G(y, \neg y) \le G(a(x), b(x))$ by monotonicity of $G$. If $x' \sqsubseteq_L x$ then
  $a(x') \le a(x)$ and $b(x') \le b(x)$, so $\overline g$ is $L$-monotone.
- $\overline g \le g_{coh}^L$: by induction in topological order, $v^+(a(x), b(x)) \le v_{coh}(x)$ for every node. At
  event edges both use $a_v(x)$, $b_v(x)$ or TRUE alike. For a gate complement $g_{coh}^L$ uses TRUE where
  $\overline g$ uses a rail value; for NOT, NAND, NOR, XNOR, `ATMOST` $g_{coh}^L$ is TRUE; for XOR,
  $P_j \le P_{j-1} \vee \ell_j$ gives $P_m \le \bigvee_j \ell_j$ by induction on the chain; `EXACTLY` and
  `cardinality` drop a conjunct; AND, OR, `ATLEAST` are monotone in their inputs, which are dominated by the induction
  hypothesis. $g_{coh}^L$ is a monotone function of $(a(x), b(x))$, hence $L$-monotone.
- $g_{coh}^L \le g_{coh}$ for $L \supseteq L^+$: $g_{coh}$ is $g_{coh}^L$ with the kept direct complements replaced by
  TRUE, and its gates are monotone. ∎

Remarks:

- If the cone has no negation at all, $g = \underline g = \omega_L g = \overline g = g_{coh}^L$. For Theorem 3's form
  with $F$ and $D$ written without gate complements, $\overline g = g_{coh}^L = F$.
- If the only negations are direct complements, $g$ is $L_g$-monotone and all five functions coincide for $L_g$: no
  lane is a candidate (§8.8, §8.9). With $L^+$ the same target has $g_{coh} > g$.
- $\underline g$, $\overline g$ and $g_{coh}^L$ depend on the circuit, not only on $g$ ($A \wedge \neg\neg B$ has
  $g_{coh} = A$, $\overline g = A \wedge B$). The estimator's correctness needs only the inequalities; tightness affects
  cost only.
- In the two-flag example (§8.2) the flags occur only under negation, so they are negative-unate and
  $\underline g = \uparrow g$; whether this holds on a given model is a measurement, not a theorem.

### 4.5 Probability-0 and probability-1 events

**Theorem 5.** Let $h = \omega_L g$ and $L'$ the literals of $L$ over $V \setminus \lbrace v \rbrace$; $\omega$ below is
$\omega_{L'}$.

- $v$ droppable: $h_{v \leftarrow 0} = \omega(g_{v \leftarrow 0})$ and $h_{v \leftarrow 1} = \omega(\exists_v g) = \omega(g_{v \leftarrow 0}) \vee \omega(g_{v \leftarrow 1})$.
- $v$ fixed: $h_{v \leftarrow c} = \omega(g_{v \leftarrow c})$ for $c \in \lbrace 0, 1 \rbrace$.
- $v$ raisable: $h_{v \leftarrow 1} = \omega(g_{v \leftarrow 1})$ and $h_{v \leftarrow 0} = \omega(\exists_v g)$.
- $v$ free: $h_{v \leftarrow 0} = h_{v \leftarrow 1} = \omega(\exists_v g)$.

*Proof.* $y \sqsubseteq_L x[v \leftarrow c]$ iff $y_v \in A_v^L(c)$ and the other coordinates satisfy
$\sqsubseteq_{L'}$. Splitting on $y_v$ gives one hull per allowed value; the disjunction is Theorem 6(a). ∎

**Corollary 5.1.** If $p_v = 0$ then $X = X[v \leftarrow 0]$ almost surely: replacing a droppable or fixed event by
FALSE preserves $P(\omega_L g)$ (a raisable one must be eliminated by $\exists_v$). If $p_v = 1$: replacing a fixed or
raisable event by TRUE preserves it; a droppable event must be eliminated by $\exists_v$, and $P(\omega_L g)$ differs
from the value with TRUE substituted whenever $\omega(g_{v \leftarrow 0}) \not\le \omega(g_{v \leftarrow 1})$ on a set of
positive probability. Substituting TRUE for a probability-1 droppable event is wrong for the hull; the correct
elimination is existential quantification: the event may be dropped.

Under Boolean semantics probability-1 events behave as constants; under the hull, droppable probability-1 events can be
dropped, and only probability-0 events and fixed events behave as constants. This matches cut-set tools, which keep
probability-1 events in cut sets and apply delete terms to them. Constants (A3) are not droppable: a house event set TRUE
or an override `X := TRUE` stays TRUE in every $y$.

### 4.6 OR, AND and modules

**Theorem 6.** (a) $\omega_L(g \vee h) = \omega_L g \vee \omega_L h$. (b) $\omega_L(g \wedge h) \le \omega_L g \wedge \omega_L h$,
with equality when $g$ and $h$ depend on disjoint sets of events, and $\omega_L(g \wedge h) = \omega_L g \wedge h$ when
$h$ depends only on fixed events. (c) Equality in (b) fails in general, even when $h$ is monotone: $g = \neg A$, $h = A$
and $L = L^+$ give $\uparrow(g \wedge h) = 0$ and $\uparrow g \wedge \uparrow h = A$.

*Proof.* (a) $\exists y \sqsubseteq_L x$ distributes over $\vee$. (b) A witness $y$ for $g \wedge h$ witnesses both.
$\sqsubseteq_L$ is a product of per-event relations, so with disjoint supports $V_g$, $V_h$, witnesses on $V_g$ and on
$V_h$ combine into one. The last claim is Lemma 1.2. ∎

Consequences:

- Per-lane hull indicators compose by OR on the same samples: the hull of an end state or accident class (an OR of
  sequences) is the OR of the per-sequence indicators, at no extra check cost.
- Hulls of mutually exclusive sequences are in general **not** mutually exclusive:
  $P(\omega_L(s_1 \vee s_2)) \le P(\omega_L s_1) + P(\omega_L s_2)$, strictly in §8.5. End-state values are their own
  queries; do not sum per-sequence hull values.
- There is no decomposition through AND: $\uparrow(F \wedge \neg D) \ne F \wedge \uparrow\neg D$, and
  $\uparrow\neg D = 1$ whenever $D(0) = 0$.

**Proposition 6.1 (modules).** Let $g(x) = G(M(x_A), x_B)$ with $A \cap B = \emptyset$, and suppose every $x_A$ has some
$y_A \sqsubseteq_L x_A$ with $M(y_A) = 0$ (for example $M(0) = 0$ and every event of $A$ droppable). Then
$P(\omega_L g) = P(\omega_{L_Z} G(Z, X_B))$, where $Z$ is an independent droppable event with $P(Z = 1) = P(\omega_L M)$.

*Proof.* For $y_A \sqsubseteq_L x_A$ the values of $M(y_A)$ include 0 by hypothesis and include 1 iff
$\omega_L M(x_A) = 1$. Hence $\omega_L g(x) = \exists y_B \sqsubseteq_L x_B : G(0, y_B) \vee (\omega_L M(x_A) \wedge G(1, y_B))$,
which is $\omega_{L_Z} G(\cdot, \cdot)$ at $(\omega_L M(x_A), x_B)$, and $\omega_L M(X_A)$ is independent of $X_B$. ∎

Without the hypothesis a module needs two indicators, $\omega_L M$ and $\omega_L \neg M$. Collapsing a module into one
`COLLAPSED_MODULE` sample with its Boolean probability $P(M)$ is therefore correct for a hull target only if
$P(\omega_L M) = P(M)$, for example when $M$ is $L$-monotone (§8.7).

### 4.7 Relation to cut-set quantification and Boolean probability

By Theorem 4 and monotonicity of $P$: $P(g) \le P(\underline g) \le P(\omega_L g) \le P(\overline g) \le P(g_{coh}^L)$,
and by Theorem 1(d), $P(\omega_{L_g} g) \le P(\uparrow g)$. The gap between $P(g)$ and $P(\omega_L g)$ is unbounded in
ratio: in the private model $P(g) = 0 < P(\uparrow g)$ (§8.2 in miniature).

A cut-set tool with delete terms computes (a truncated part of) the family $K = K_h$ of Theorem 10(a), atoms read as
literals; under Theorem 10(c)'s hypothesis it has the same disjunction as $\operatorname{MC}_{L_g}(g)$, and equals it
when no event occurs in it with both polarities. For the `omega_l` option, $K = \operatorname{MC}_L(g)$ (Theorems 2 and
3). It quantifies $K$ with the rare-event approximation $\mathrm{REA} = \sum_{C \in K} P(C)$ or the min-cut
upper bound $\mathrm{MCUB} = 1 - \prod_{C \in K}(1 - P(C))$, with $P(C)$ as in §2.

**Proposition 7.** Under A2, without truncation, for the hull $\eta \in \lbrace h, \omega_L g \rbrace$ whose cut sets
$K$ are, $P(\eta) \le \mathrm{REA}$. If no event occurs with both polarities among the members of $K$ (always the case
for $L = L^+$), $P(\eta) \le \mathrm{MCUB} \le \mathrm{REA}$.

*Proof.* $\eta = \bigvee_{C \in K} C$ on worlds (Theorem 2(b), Theorem 10(a)), so the union bound gives the first
claim. Under the
hypothesis replace $X_v$ by $1 - X_v$ for every event occurring only negatively; the variables stay independent and
every event $\lbrace X \text{ satisfies } C \rbrace$ becomes increasing, its complement decreasing. Independent
variables are associated, and decreasing functions of associated variables are positively correlated [EPW67], so
$P(\bigcap_C \lbrace X \text{ does not satisfy } C \rbrace) \ge \prod_C (1 - P(C))$; this is the Esary–Proschan
min-cut bound [EP63]. $1 - \prod(1 - a_i) \le \sum a_i$ for $a_i \in [0, 1]$ by induction. ∎

REA can exceed 1 when probability-1 events occur in cut sets (§8.1).

**Truncation.** Tools drop cut sets below a probability cutoff (or above a size). If cut sets of $D$ are truncated at
$\tau_D$ and cut sets of $F$ at $\tau_F$ with $\tau_D \le \tau_F$, no surviving $C \in \operatorname{MC}_L(F)$ escapes
deletion: a truncated $C' \in \operatorname{MC}_L(D)$ with $C' \subseteq C$ has $P(C) \le P(C') < \tau_D \le \tau_F$, so
$C$ is truncated too (for size cutoffs the condition is that $D$'s size limit is at least $F$'s, since
$|C| \ge |C'|$). The tool then quantifies $K_\tau = \lbrace C \in K : P(C) \ge \tau_F \rbrace$, and
$P(\omega_L g) - P(\exists C \in K_\tau : X \text{ satisfies } C) \le \sum_{C \in K \setminus K_\tau} P(C)$. If
$\tau_D > \tau_F$, deletions can be missed. A tool's number is therefore neither an upper nor a lower bound on
$P(\omega_L g)$ in general: truncation lowers it, REA and MCUB raise it. Generating cut sets with negations dropped
(success branches ignored) gives the cut sets of $\overline g$ for $L^+$ and overestimates $P(\uparrow g)$ (§8.4).

**Complexity.** For monotone $g$, $\omega_L g = g$ for every $L \supseteq L^+$, so computing $P(\omega_L g)$ exactly
includes monotone network reliability, which is #P-hard [V79]. At scale the choice is between sampling with a stated
interval and cut-set approximations of unknown sign; this RFC specifies the former.

### 4.8 The initiator rule

Quantification config option `max_initiators_per_cutset = m` (unset: no rule; the vendor tools apply $m = 1$). Let $I$
be the initiator-tagged events in the cone of the target (`I` in `IMPORT`; a probability-1 initiator flag is an event,
not a constant, and counts). The lowering replaces the target $g$ by

$$g_m = g \wedge \neg\,\mathrm{atleast}(m+1;\ I),$$

one cardinality gate, omitted when $|I| \le m$. Under Boolean semantics this is the literal conjunction; under the hull
it is a delete term.

**Corollary 3.4 (the rule).** If no initiator has a significant negative literal ($I \subseteq P_L$; for $L_g$: no
initiator has a marked literal), then
$\omega_L g_m = \bigvee \lbrace C \in \operatorname{MC}_L(g) : |C \cap I| \le m \rbrace$, where $|C \cap I|$ counts the
positive initiator literals of $C$. If moreover no event occurs with both polarities among these $C$,
$\operatorname{MC}_L(g_m)$ is exactly this family: the rule deletes the minimal cut sets with more than $m$ initiators
and nothing else.

*Proof.* $D = \mathrm{atleast}(m+1; I)$ is monotone in the droppable events $I$ and independent of all others, hence
$L$-monotone and independent of $B_L$; $\operatorname{MC}_L(D) = \operatorname{PI}(D)$ is the family of
$(m+1)$-subsets of $I$. By Corollary 3.1, $C \in \operatorname{MC}_L(g)$ is kept iff it contains no $(m+1)$-subset of
$I$. The last claim is the last claim of Theorem 3 applied to $F = \omega_L g$. ∎

Remarks:

- Initiators outside the cone do not matter under the hull: setting such an event to 0 in a witness keeps it a witness,
  and that event cannot raise the count. Under Boolean semantics they would, which is why $I$ is taken in the cone.
- Under Boolean semantics a probability-1 initiator flag makes the rule remove every world in which another initiator
  occurs; under the hull it removes only the cut sets that contain both (§8.10).
- The vendor configuration declares the rule per quantification engine (§1), so it is a config option, not a
  model property. It has no default and is never read from the model (#262); the lowering applies it to hull targets
  only, since under Boolean semantics it changes the target.

### 4.9 The atom-split hull

The atom-split hull is the plain hull $\uparrow\tilde g$ of §§4.1–4.8 in atom space, read at $\kappa(x)$; every
statement below follows from those results applied to $\tilde g$ with $L^+$ over $\mathcal{A}$, except where the
coupling of $v$ and $\bar v$ enters.

**Theorem 10.** Let $\Delta$, $\tilde g$, $\kappa$, $h$ and $K_h$ be as in §§2–3, and $L = L_g$.

- (a) *Cut sets.* $h(x) = 1$ iff $\kappa(x)$ satisfies some member of $K_h$. If $\tilde g = \tilde F \wedge \neg\tilde D$
  with $\tilde F$, $\tilde D$ monotone over atoms (sequences whose negations are gate complements), then
  $\operatorname{mcs}(\uparrow\tilde g) = \lbrace C \in \operatorname{mcs}(\tilde F) : \text{no } C' \in \operatorname{mcs}(\tilde D) \text{ has } C' \subseteq C \rbrace$:
  delete-term processing over atoms, followed by removing the sets that contain an event and its complement atom,
  yields exactly $K_h$. $K_h$ is an antichain of atom sets and contains no consensus term.
- (b) *Bounds.* $g \le \omega_L g \le h \le g_{coh}^L$.
- (c) *Agreement with $\omega_L$.* If every reference to an event of $\Delta$, direct or complemented, is reached from
  the target through an even number of gate complements (in particular if no event of $\Delta$ occurs in the cone of a
  complemented gate), then $h = \omega_L g$; if moreover no event has both atoms in members of $K_h$, then $K_h$, with
  $\bar v$ read as $\neg x_v$, is $\operatorname{MC}_L(g)$. Otherwise $h$ and $\omega_L g$ can differ.
- (d) *Not below the plain hull.* $h \le \uparrow g$ fails in general; under (c)'s hypothesis $h = \omega_L g \le \uparrow g$.
- (e) *Estimator.* On every world,
  $g(x) \le \underline{\tilde g}(\kappa(x)) \le h(x) \le \overline{\tilde g}(\kappa(x)) \le \tilde g_{coh}(\kappa(x)) = g_{coh}^L(x)$,
  with the bounds of Theorem 4 for $L^+$ on the split circuit ($\tilde g_{coh}$: gate negations relaxed, atoms kept),
  and $h(x)$ is the per-lane check of §6 with $L^+$ on the split circuit at $\kappa(x)$, whose true atoms are
  $T(x) \cup \lbrace \bar v : v \in \Delta,\ x_v = 0 \rbrace$.
- (f) *Initiator rule.* With $g_m = g \wedge \neg\mathrm{atleast}(m+1; I)$ (§4.8), $h_m(x) = 1$ iff $\kappa(x)$ satisfies
  some $C \in K_h$ with at most $m$ positive initiator atoms: the rule deletes exactly the atom cut sets with more than
  $m$ initiators, and complement atoms of initiators do not count. With no initiator in $\Delta$ this is the rule of
  Corollary 3.4.
- (g) *Probability-0 and probability-1 events.* An atom that is 0 almost surely (the event atom of a $p = 0$ event, the
  complement atom of a $p = 1$ event) may be replaced by FALSE. An atom that is 1 almost surely (the complement atom
  $\bar v$ of a $p_v = 0$ event, the event atom of a $p = 1$ event) is droppable: replacing it by TRUE preserves $P(h)$
  if it is positive-unate in $\tilde g$ (in particular if it does not occur under a gate complement), and not in
  general.
- (h) *Coupling.* Factorisations need both atoms of an event on the same side. If $\tilde g = G(M(a_A), a_B)$ with
  disjoint atom sets $A$, $B$, no event having one atom in $A$ and the other in $B$, and $M(0) = 0$, then Proposition
  6.1 holds in atom space: $P(h) = P(\uparrow G(Z, \kappa(X)_B))$ with $Z$ an independent droppable event,
  $P(Z = 1) = P(\uparrow M(\kappa(X)_A))$. Without the condition $\uparrow M(\kappa(X)_A)$ and $\kappa(X)_B$ are
  dependent. Likewise Proposition 7's min-cut bound needs that no event has both atoms in members of $K_h$.

*Proof.* (a) $\uparrow\tilde g$ is monotone over atoms, so it is the disjunction of its minimal cut sets (Theorem 2(d)
with $L^+$), and $\kappa(x)$ satisfies no set containing $\lbrace v, \bar v \rbrace$. The second claim is Theorem 3 with
$L^+$ over atoms ($B_{L^+} = \emptyset$, so its hypothesis on $D$ is void). A consensus term would need a variable with
both polarities in two members; members of $K_h$ are positive over atoms.
(b) A witness $y$ of $\omega_L g(x)$ has $y_\Delta = x_\Delta$, so $\tilde g(y, \kappa(x)_{\bar\Delta}) = g(y) = 1$ with
$(y, \kappa(x)_{\bar\Delta}) \le \kappa(x)$: it witnesses $h(x)$. $\tilde g \le \tilde g_{coh}$ and $\tilde g_{coh}$ is
monotone (Theorem 4 with $L^+$), so $\uparrow\tilde g \le \tilde g_{coh}$ (Theorem 1(b)); $\tilde g_{coh}$ keeps the
atoms $\bar v$, which $g_{coh}^L$ keeps as the literals $\neg x_v$, so $\tilde g_{coh} \circ \kappa = g_{coh}^L$.
(c) Under the hypothesis the atoms $v \in \Delta$ and $\bar v$ are positive-unate in $\tilde g$ (§4.4), so a witness
$b \le \kappa(x)$ of $h(x)$ stays a witness when $b_v$ and $b_{\bar v}$ are raised to $\kappa(x)_v$ and
$\kappa(x)_{\bar v}$ for every $v \in \Delta$; the result is $\kappa(y)$ for the world $y$ that equals the raised valuation
on $V$, and $y \sqsubseteq_L x$ with $g(y) = \tilde g(\kappa(y)) = 1$. With (b), $h = \omega_L g$. If no event has both atoms in $K_h$, read as $L$-terms its members form an
antichain on which step 4 of Theorem 2 applies: $\operatorname{PI}(\bigvee K_h) = K_h$, and
$\operatorname{MC}_L(g) = \operatorname{PI}(\omega_L g) = K_h$. The probes below differ.
(d) The second probe below: $h = A$, $\uparrow g = AB$. The second claim is Theorem 1(d).
(e) Theorem 4 with $L^+$ holds pointwise on $\lbrace 0,1 \rbrace^{\mathcal{A}}$, in particular at $\kappa(x)$; the
check decides $\uparrow\tilde g$ at any valuation (§6.1). The equality is (b).
(f) $D = \mathrm{atleast}(m+1; I)$ is monotone over atoms. Corollary 3.1 with $L^+$ over atoms gives
$\uparrow(\tilde g \wedge \neg D) = \bigvee \lbrace C \in \operatorname{mcs}(\uparrow\tilde g) : |C \cap I| \le m \rbrace$;
read at $\kappa(x)$ as in (a).
(g) Theorem 5 (droppable case) in atom space: for an atom $a$, $(\uparrow\tilde g)_{a \leftarrow 0} = \uparrow(\tilde g_{a \leftarrow 0})$
and $(\uparrow\tilde g)_{a \leftarrow 1} = \uparrow(\tilde g_{a \leftarrow 0}) \vee \uparrow(\tilde g_{a \leftarrow 1})$. On
the worlds of positive probability the atom has its almost-sure value. If $a$ is positive-unate,
$\tilde g_{a \leftarrow 0} \le \tilde g_{a \leftarrow 1}$, so the disjunction is $\uparrow(\tilde g_{a \leftarrow 1})$. The
counterexample: $A \wedge \neg(A \wedge \neg B)$ with $p_B = 0$ has $\tilde g = A \wedge \neg(A \wedge \bar B)$ and $h = A$;
with $\bar B$ replaced by TRUE it is 0.
(h) As in Proposition 6.1, $h(x) = \uparrow G(\uparrow M(\kappa(x)_A), \kappa(x)_B)$; $\uparrow M(\kappa(X)_A)$ is a function
of the events with atoms in $A$, independent of those with atoms in $B$ under the condition. For the min-cut bound,
the events $\lbrace \kappa(X) \text{ satisfies } C \rbrace$ are increasing in the independent events after flipping the
events that occur only through $\bar v$, as in Proposition 7. ∎

Without (h)'s condition the min-cut bound can fail: $K_h = \lbrace \lbrace A, B \rbrace, \lbrace C, \bar B \rbrace \rbrace$
($g = AB \vee C\neg B$) with $p_A = p_C = 1$, $p_B = 0.5$ has $P(h) = 1$ and
$\mathrm{MCUB} = 1 - 0.5 \cdot 0.5 = 0.75$. REA is always an upper bound.

**Vendor agreement.** On every conformance fixture both vendor tools list exactly $K_h$ above the fixture's `LIMIT`
(with the initiator rule; and without it once the `I` tags are removed), including `complement_under_negation`
(§8.11), where $\omega_{L_g}$ and the plain hull do not. Probe models:

| probe | vendor tools = $K_h$ | $\operatorname{MC}_{L_g}(g)$ | $\operatorname{mcs}(\uparrow g)$ |
|---|---|---|---|
| $A \wedge \neg(B \vee E) \;\vee\; D \wedge \neg B$ | $\lbrace A \rbrace$, $\lbrace \bar B, D \rbrace$ | $\lbrace A, \neg B \rbrace$, $\lbrace D, \neg B \rbrace$ | $\lbrace A \rbrace$, $\lbrace D \rbrace$ |
| $A \wedge \neg(A \wedge \neg B)$ | $\lbrace A \rbrace$ | $\lbrace A, B \rbrace$ | $\lbrace A, B \rbrace$ |
| $AB \vee C \wedge \neg B$ | $\lbrace A, B \rbrace$, $\lbrace \bar B, C \rbrace$ | adds $\lbrace A, C \rbrace$ | $\lbrace A, B \rbrace$, $\lbrace C \rbrace$ |
| $A \wedge \neg G$, $G$ a single-input OR (or AND) of $B$ | $\lbrace A \rbrace$ | $\lbrace A \rbrace$ | $\lbrace A \rbrace$ |

With $p_A = 0.1$, $p_B = 0.2$, $p_D = 0.3$, $p_E = 0.4$, the first probe has $P(h) = 0.1 + 0.9 \cdot 0.3 \cdot 0.8 = 0.316$
against $P(\omega_{L_g} g) = 0.296$; the second has $P(h) = p_A$. In the first, no $L$ at all makes $\omega_L g$ equal
to $h = A \vee D\neg B$ (checked over all $4^4$ choices): the tools' rule is not a widening of $g$. In the third,
$\lbrace A, C \rbrace$ is not an implicant over atoms ($A = C = 1$ with both $B$ and $\bar B$ dropped). The last row: a
complemented single-input gate is a delete term, not an atom.

## 5. Estimator

§5 and §6 are written for $\omega_L g$, which includes the plain hull. For the atom-split hull they apply
verbatim to $\uparrow\tilde g$ with $L = L^+$ over atoms, evaluated at $\kappa(X_k)$ (Theorem 10(e)): lanes draw events,
the complement atoms are their complements (one draw per event), the bounds are those of the split circuit, and the
check runs on the split circuit. In the estimates below read $\omega_L g$ as $h$.

### 5.1 Per-lane evaluation

Draw $N$ independent worlds ("lanes") $X_1, \dots, X_N$. For each lane evaluate $\underline g$ and $\overline g$
bit-parallel (64 lanes per word, or any width), then set

- $H_k = 0$ if $\overline g(X_k) = 0$;
- $H_k = 1$ if $\underline g(X_k) = 1$;
- otherwise ($\underline g = 0$, $\overline g = 1$: a *candidate* lane) $H_k = \omega_L g(X_k)$, decided exactly by the
  check in §6.

The estimate is $\hat\theta = N^{-1}\sum_k H_k$. Using $g$ and $g_{coh}^L$ in place of $\underline g$ and $\overline g$
is also correct; the tighter bounds only reduce the number of candidates. The device produces a candidate mask per
word; the host gathers the event bits of candidate lanes, runs the checks in parallel, and adds integer counts.

### 5.2 Unbiasedness, variance, intervals

By Theorem 4, $H_k = \omega_L g(X_k)$ on every lane, so the $H_k$ are i.i.d. Bernoulli($\theta$) with
$\theta = P(\omega_L g)$: $E\hat\theta = \theta$ and $\operatorname{Var}\hat\theta = \theta(1 - \theta)/N$.
Intervals: Wilson score [W27] by default; Clopper–Pearson [CP34] when hits are few. With zero hits the two-sided 95%
Clopper–Pearson upper limit is $1 - 0.025^{1/N} \approx 3.69/N$.

Decomposition. With $a = P(\underline g = 1)$, $q = P(\underline g = 0, \overline g = 1)$ and
$\rho = P(\omega_L g = 1 \mid \underline g = 0, \overline g = 1)$, $\theta = a + q\rho$ and
$\hat\theta = (N_L + N_H)/N$, where $N_L$ counts lanes with $\underline g = 1$ and $N_H$ candidate lanes whose check
succeeds. The product form of #174, $\theta = \pi\rho'$ with $\pi = P(\overline g = 1)$ and
$\rho' = P(\omega_L g \mid \overline g = 1)$, gives the same number ($\hat\pi\hat\rho' = N_U/N \cdot (N_L + N_H)/N_U$,
$N_U$ counting lanes with $\overline g = 1$). Its delta method variance is
$\rho'^2\pi(1 - \pi)/N + \pi^2\rho'(1 - \rho')/(N\pi) = \theta(1 - \theta)/N$ (the factors are asymptotically
uncorrelated because $E[\hat\rho' \mid N_U] = \rho'$): the product form is useful for reporting, not for a different
interval.

Per-run bounds: $\hat\theta_L = N_L/N$ and $\hat\theta_U = N_U/N$ are unbiased for $P(\underline g)$ and
$P(\overline g)$ and satisfy $\hat\theta_L \le \hat\theta \le \hat\theta_U$ on every run, since the inequalities hold
lane by lane.

### 5.3 Subsampled checks

When checks dominate cost, check each candidate lane independently with probability $r \in (0, 1]$ and weight it by
$1/r$. With selection bits $S_k \sim$ Bernoulli($r$) independent of $X$, and $\underline g_k$, $\overline g_k$ the bound
values of lane $k$:
$\tilde H_k = \underline g_k + (1 - \underline g_k)\overline g_k S_k\, \omega_L g(X_k)/r$.

- Unbiased: $E[\tilde H_k \mid X_k] = \omega_L g(X_k)$.
- Variance: $\operatorname{Var}\tilde H_k = \theta(1 - \theta) + \delta(1/r - 1)$, with
  $\delta = P(\underline g = 0, \omega_L g = 1)$. (Derivation: $\underline g$ and the candidate term are disjoint
  indicators, so $E\tilde H^2 = a + \delta/r$.)
- Interval: $\hat\theta \pm z\,s/\sqrt N$ with $s^2$ the sample variance of $\tilde H$. With few candidate hits the
  normal approximation is poor; use $r = 1$ there.
- Efficiency. Let $c_1$ be the per-lane cost of the bit-parallel stage and $c_2$ the mean cost of one check. The work
  per lane is $c_1 + rqc_2$ and the work-normalised variance is $(c_1 + rqc_2)(V_0 + \delta/r)$ with
  $V_0 = \theta(1 - \theta) - \delta = a(1 - a - 2\delta) - \delta^2$. If $V_0 \le 0$, for example when $a = 0$ and every
  hit comes from a check (the prototype's setting: $\underline g = g$, which is identically 0 on the private model's
  literal target), $r = 1$ is optimal: subsampling is dominated by running fewer lanes. If $V_0 > 0$, the optimum is
  $r^\star = \min(1, \sqrt{c_1\delta/(qc_2V_0)})$ (§8.4: 1.5× less work at $c_2 = 100c_1$).
- Alternative (two-phase): per pass, check a simple random sample of $m$ of the $N_c$ candidates with weight $N_c/m$.
  Conditionally on the lanes this is unbiased, and its extra variance is about $q\rho(1 - \rho)(1 - r)/(rN)$ instead
  of $q\rho(1 - r)/(rN)$: smaller by the factor $1 - \rho$, which matters only when most candidates satisfy the hull.

### 5.4 Cost model

The bit-parallel stage evaluates the cone for $\underline g$ and $\overline g$. $\underline g$ shares every node that
does not depend on a unate droppable, raisable or free event; $\overline g$ shares every node whose cone has no literal
outside $L$. Both add at most one copy of the respective region, and $g$ itself need not be evaluated. Sampling cost
follows edges (#177).

Expected cost per lane is $c_1 + qc_2$ with $q = P(\underline g = 0, \overline g = 1) \le P(g_{coh}^L) - P(g)$. The
prototype, using $g$ and $g_{coh}$, had $q \approx 3.6\%$. With $\underline g$, $q$ is model-dependent (0 in §8.2,
§8.8, §8.9).

Check cost: after the fixing of §6.2 the free events of a lane are binate events whose allowed set has two values:
droppable ones in $T(x)$, raisable ones outside $T(x)$, and free ones. Fixed events are never free. With $\mathrm{Bin}$
the binate events, enumeration needs at most $2^{\#\text{free}}$ residual evaluations (64 per word), and by
independence
$E[2^{\#\text{free}}] = \prod_{v \in \mathrm{Bin} \cap P_L}(1 + p_v) \prod_{v \in \mathrm{Bin} \cap N_L}(2 - p_v) \prod_{v \in \mathrm{Bin} \cap O_L} 2$
(since $E[2^{X_v}] = 1 + p_v$). Each binate droppable probability-1 event doubles this bound; that is the failure mode
to watch.

### 5.5 Importance sampling

With a product proposal $Q$ (event probabilities $\tilde p_v$) and weight $w(x) = \prod_v P(X_v = x_v)/Q(X_v = x_v)$, the
estimator $N^{-1}\sum_k w(X_k)H_k$, $X_k \sim Q$, is unbiased for $\theta$ provided $\tilde p_v \in (0, 1)$ whenever
$p_v \in (0, 1)$ and $\tilde p_v = p_v$ when $p_v \in \lbrace 0, 1 \rbrace$. $\omega_L g$ is a deterministic function of
the lane, so the check, the bounds and the weights are unchanged: the weight multiplies whatever $H_k$ (or $\tilde H_k$)
the lane produced (#6). The variance is $V_Q/N$, with $V_Q$ the variance of $wH$ under $Q$, estimated by the sample
variance; use the unnormalised form. The candidate fraction under $Q$, $Q(\underline g = 0, \overline g = 1)$, is
typically larger than under $P$ when $Q$ raises initiator probabilities, so checks per lane increase.

### 5.6 Determinism

- Lanes are reproducible from (seed, sample index) with counter-based streams per pcfb SPEC §4.
- $H_k$ is a deterministic function of the lane and counts are integers, so $\hat\theta$ is identical for any thread
  count, check order or device that enumerates the same sample indices.
- Subsampling selection is keyed by (seed, global sample index) in a key domain disjoint from the event streams.
- Budgets that choose between check algorithms are counted in deterministic units (BDD nodes, solver conflicts), never
  wall-clock time; they change cost, not answers (§6.4).

## 6. The per-lane check

### 6.1 Problem and complexity

Given a circuit, a target, $L$ and a world $x$ with true set $T$, decide $\omega_L g(x)$: is $g$ satisfiable with every
event $v$ restricted to $A_v^L(x_v)$? For $L^+$: every event outside $T$ fixed to 0 and every event in $T$ free. For the
atom-split hull: the $L^+$ check on the split circuit at $\kappa(x)$, whose free atoms are the true events and $\bar v$
for the false events $v \in \Delta$ (Theorem 10(e)).

**Theorem 8.** For every $L$ the check is in NP. It is NP-complete already for $L = L^+$, and stays NP-hard when
$g = F \wedge \neg D$ with $F$, $D$ monotone and $D$ a disjunction of pairwise conjunctions (pairwise exclusions), for
$x = 1_V$.

*Proof.* A witness $y$ is checkable in linear time. Hardness, by reduction from CNF-SAT [C71]: for a CNF $\varphi$ over
$u_1, \dots, u_m$ introduce events $u_i^+$, $u_i^-$; let $F$ be $\varphi$ with $u_i \mapsto u_i^+$ and
$\neg u_i \mapsto u_i^-$, conjoined with $\bigwedge_i (u_i^+ \vee u_i^-)$, and let $D = \bigvee_i (u_i^+ \wedge u_i^-)$.
$F$ and $D$ are monotone and have no direct complement, so $L_g = L^+$. A $y$ with $F(y) = 1$ and $D(y) = 0$ makes
exactly one of $u_i^+, u_i^-$ true for each $i$, which encodes an assignment satisfying $\varphi$, and every satisfying
assignment gives such a $y$. ∎

**Theorem 9.** Unless P = NP, no polynomial-time algorithm maps a circuit for $g$ (and $L$) to a circuit for
$\omega_L g$.

*Proof.* For $L = L^+$, evaluating the output circuit at $1_V$ would decide CIRCUIT-SAT in polynomial time, since
$\uparrow g(1_V) = 1$ iff $g$ is satisfiable. ∎

Theorem 9 is why the hull is a query kind evaluated per sample, not a compile-time rewrite to a Boolean target.

### 6.2 Algorithm

Once per circuit and target: the cone, a topological order, fan-out lists, the class of every event under $L$, and the
polarity (positive-unate, negative-unate, binate) of every event from the dual rail of §4.4.

Per candidate lane:

1. Fix every event with one allowed value: droppable events outside $T$ to 0, raisable events in $T$ to 1, fixed events
   to their value in $x$.
2. Fix every remaining (free) positive-unate event to 1 and every free negative-unate event to 0. *Lemma:* if some
   $y \sqsubseteq_L x$ satisfies $g$, then raising a free positive-unate coordinate to 1 or lowering a free
   negative-unate coordinate to 0 keeps $g(y) = 1$ and $y \sqsubseteq_L x$, since both values are allowed. (This is
   the pure-literal rule lifted to circuits.)
3. Propagate constants: AND with a 0 input is 0, OR with a 1 input is 1, cardinality bounds shift by the number of
   inputs fixed to 1 and the arity drops by the number fixed, complemented edges and XOR parity fold.
4. If the target is constant, return it. Otherwise optionally recompute polarity on the residual (events can become
   unate after propagation) and repeat 2–3 to a fixpoint.
5. Decide satisfiability of the residual over its free events (binate events with two allowed values): enumeration
   when there are at most $k_{enum}$ free events (16–20; 64 assignments per word), else a BDD under a node budget, else
   complete search (DPLL/CDCL on a Tseitin encoding of the residual), which always terminates.

Every branch returns $\omega_L g(x)$ exactly. Lanes are independent and run in parallel (#178).

### 6.3 Why residuals are small

- $E|T \cap \text{cone}| = \sum_v p_v$: apart from high-probability and probability-1 events, few events are true.
- After step 1 every OR whose inputs are all false collapses, and with it most of a fault-tree cone; every fixed event
  is a constant.
- After step 2 only binate events stay free. In delete-term models these are events that occur both in $F$'s cone and
  in some $S$ or $DEL$. Pairwise exclusions among high-probability events are of this kind; flags that occur only under
  negation are not.
- The expected enumeration work is bounded as in §5.4.

When many shared high-probability events are binate, residuals grow and the BDD and complete-search tiers do the work;
Theorem 8 says no algorithm avoids this in the worst case.

### 6.4 Budgets and fallback

Budgets (BDD nodes, solver conflicts) select the next algorithm; the last tier is complete and has no budget. An
optional hard cap per lane (off by default) marks a lane *unresolved*. With $u$ unresolved lanes the run is flagged
incomplete and reports $[N_{hit}/N, (N_{hit} + u)/N]$ widened by the interval of §5.2; unresolved lanes are never
counted as 0 or 1.

### 6.5 Exact evaluation for small models

For up to about 24 variables: tabulate $g$ over all worlds, then close the table along each event axis, $n2^n$ operations
in all: for a droppable $v$, for each $x$ with $x_v = 1$, $H[x] \leftarrow H[x] \vee H[x \oplus 1_{\lbrace v \rbrace}]$
(the subset-OR or zeta transform when every event is droppable); for a raisable $v$ the same for $x_v = 0$; for a free
$v$ both; nothing for a fixed $v$. The result is $\omega_L g$ because $\sqsubseteq_L$ is a product of per-event
relations, and $P(\omega_L g) = \sum_x H[x]P(X = x)$. Probability-0 droppable or fixed events may be substituted by 0,
probability-1 fixed or raisable events by 1 (Corollary 5.1); other events must stay variables. For the atom-split hull
tabulate $\tilde g$ over all atom assignments, apply the subset-OR transform over every atom, and weight an assignment
$a$ by $\prod_{\text{events}} P(X_v = a_v)$ if $a_{\bar v} = 1 - a_v$ for every $v \in \Delta$ and by 0 otherwise.
pcfb's test-only `exact_hull_probability` implements the $L^+$ case; the reference tools of #179 implement all three
modes and list $K_h$ and $\operatorname{MC}_L(g)$ (#190).

## 7. Interfaces

### 7.1 `ir::pra` quantification config (#181)

- Per target: `negation: boolean | monotone_hull`, default `boolean`. Allowing both on one target is recommended: the
  writer then emits two queries on one tensor (for #185, the canonical hull target and the degeneracy check in one
  file).
- Per hull target: the interpretation of marked literals, `atom_split | omega_l | negation`, default `atom_split`
  (§3). `omega_l` and `negation` are explicit overrides, for comparisons.
- `max_initiators_per_cutset: integer ≥ 0`, default unset: a cut-set rule of hull queries, lowered per hull target as
  the delete term of §4.8 over the initiator-tagged events in the target's cone. Boolean targets never get it; a
  target with both kinds gets the Boolean query on the target and the hull query on the conjunction (#262). #185's
  canonical target is `monotone_hull`, `atom_split`, `max_initiators_per_cutset: 1`.
- The rule is not a source convention either (#262): an earlier revision had the FTP reader record
  `ftp.max_initiators_per_cutset = 1` for every project and the compile driver turn it into a default for every
  target, Boolean ones included.
- The query type is not a source convention (#244). Every target is a probability query unless the request asks for
  the hull, whatever the model and the target's cone; the compiler then chooses passes by query type (§7.5). An
  earlier revision had the FTP reader record `ftp.negation = delete_term` and made the hull the FTP default.
- Overrides `element := TRUE | FALSE` produce constants under both semantics; a basic event with literal $p = 1$ stays
  an event. The lowering is function-preserving (#181) and must keep this distinction: house events and overrides
  become constants or `house_event` nodes, basic events with $p \in \lbrace 0, 1 \rbrace$ stay `basic_event` nodes.
- The lowering does not infer literals. $\Delta$ is the set of events with a marked operand (`literal` flag) in the
  target's cone; unmarked complements, from any source, are ordinary negations. The lowering keeps every complemented
  basic-event operand as a complemented edge on the basic-event node, exactly as in the source, and never turns a gate
  complement into a complemented event edge (a complemented single-input gate stays a complemented gate). The kind, the
  literal mode and $\Delta$ are recorded per pc target on the `pra_pc_lowering` artifact, next to the target bindings;
  the initiator rule is ordinary logic in the lowered target. An event with both marked and unmarked complemented
  operands in one target's cone cannot be represented by the per-sample list of §7.3 and gets a diagnostic; the FTP
  reader never produces it.
- For each marked event that also occurs in the cone of a complemented gate of the target (or is an initiator under the
  initiator rule), the lowering emits a warning with the stable id `complement-under-negation`: the atom-split hull and
  $\omega_{L_g}$ can differ there (Theorem 10(c)).

### 7.2 `ir::pc`: unchanged

Targets do not encode result kinds (`docs/IR-PC-REQUIREMENTS.md` §2). The kind, the literal mode and $\Delta$ travel
from the config to the lowering artifact, to the writer options (a target → kinds map, like the symbol maps of #176), to `.pcfb`
`Query.kind` (§7.3). A `.pcfb` → `ir::pc` reader returns them in a side table keyed by target; a consumer that cannot
carry it must diagnose non-`PROBABILITY` queries, not drop them.

### 7.3 `.pcfb`: `QueryKind.MONOTONE_HULL_PROBABILITY` (pcfb 1.2)

- **Schema.** `MONOTONE_HULL_PROBABILITY = 1` appended to `QueryKind`; wire-compatible; `flatc --conform` against the
  frozen v1 and v1.1 schemas.
- **Semantics.** §3 over the `SAMPLE` outputs, as merged with $L = L^+$ (pcfb SPEC v1.2 §5). `BITWISE_ZEROS` /
  `BITWISE_ONES` are constants; `always_true` samples are droppable events.
- **Literal mode (amendment to 1.2).** pcfb#10 is merged but untagged; this RFC proposes two optional per-query fields
  for hull queries, to be added to 1.2 before it is tagged:
  - `literal_mode`: `ATOM_SPLIT` (default) | `OMEGA_L` | `NEGATION`;
  - `literal_samples`: indices of the `SAMPLE` outputs of $\Delta$ (the events with marked literals). Under
    `ATOM_SPLIT` every complemented edge into a listed sample, within the query's target cone, reads that sample's
    complement atom (droppable, valued $\neg x$); under `OMEGA_L` the listed samples are fixed; under `NEGATION` the
    list is ignored. Complemented edges into unlisted samples are Boolean complements in every mode.

  An absent or empty list gives the plain hull of 1.2 as merged, so existing files and readers are unaffected; a
  runtime that does not implement a mode rejects a query that uses it before sampling. Rejected alternative: deriving
  $\Delta$ from the complemented edges into `SAMPLE` nodes. Only the source knows which complements are literals, and
  passes that preserve $g$ change those edges (eliminating a
  single-input gate turns a gate complement into a complemented `SAMPLE` edge; negation normalisation creates
  complemented `SAMPLE` edges), which would change the atoms (§7.5).
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

Recommendation: the runtime derives $\underline g$ and $\overline g$ from the target's cone and $L$ at planning time with
the rules of §4.4; the writer emits no auxiliary query and no query-to-query link. Reasons:

1. **Correctness without trust.** The estimator is unbiased only if $\underline g \le \omega_L g \le \overline g$.
   Derived by fixed rules, the inequalities hold by construction (Theorem 4). A writer-supplied bound $h$ cannot be
   checked cheaply: deciding $\uparrow g \le h$ is coNP-complete (its complement,
   $\exists x, y : y \le x,\ g(y),\ \neg h(x)$, is in NP and contains SAT via $h = 0$). A wrong bound would bias the
   estimate silently.
2. **Runtime neutrality** (pcfb SPEC §1, goal 3). Bounds belong to one estimator. Exact runtimes (enumeration,
   knowledge compilation) do not use them; the file states what to compute.
3. **Size.** An auxiliary cone duplicates the negation-dependent part of the target cone in the file and in device
   memory; the runtime shares nodes and evaluates $\underline g$ instead of $g$.
4. **No schema surface.** No link field, no validation rule tying two queries together, nothing for transforms to keep
   consistent.
5. **Tightness can improve without a format change.** Writers that want tighter bounds improve the representation
   (for example negation-normal form), which the runtime exploits automatically.

Cost of the recommendation: every hull-capable runtime implements the derivation (one linear pass for polarity plus a
second rail for the negation-dependent nodes), and the bound depends on the representation, which affects cost only.
An optional writer hint would be worth adding only if a measurement shows the bounds loose on real models and a writer
can produce provably tighter bounds (§9).

### 7.5 Transform compatibility

Passes declare a semantic class (#177):

- `function_preserving`: $g' = g$ on every world. Preserves `PROBABILITY` and the `omega_l` and `negation` hulls
  ($\omega_L g$ depends only on $g$ and on $L$, a set of events fixed at lowering). It may change the bounds, hence
  cost.
- *Atom-preserving* (needed for `atom_split` targets): $\tilde g' = \tilde g$ over atoms, with the same $\Delta$, which
  is stricter than function-preserving. It forbids folding a complemented single-input gate into a complemented edge
  on an event of $\Delta$ (a delete term would become an atom), creating or removing complemented edges into events of
  $\Delta$, and complementary-pair rules on such events ($x \vee \neg x \to$ TRUE under a negated gate changes the
  result: over atoms $\neg(B \vee \bar B)$ holds when both are dropped). The simplest compliant pass leaves the nodes
  that reach a complemented edge into an event of $\Delta$ alone and says so in its transform report (#177).
- `distribution_preserving`: the target's value is unchanged under the file's independent distribution, for a stated
  set of kinds.

Recommendation: a `distribution_preserving` report lists the kinds it preserves; the planner applies a pass to a cone
only if the pass is function-preserving or every query kind in that cone is in the list. For R1, with targets of both
kinds sharing a cone, apply a pass only if it is acceptable for every kind present.

| Pass | Class | `PROBABILITY` | `MONOTONE_HULL_PROBABILITY` | Reason |
|---|---|---|---|---|
| `propagate_constants`, `eliminate_single_input`, `coalesce` (#177) | function | yes | yes, atom-preserving on `atom_split` targets (see above) | $g$ (and $\tilde g$) unchanged |
| negation normalisation | function | yes | `omega_l`, `negation`; `atom_split` only away from $\Delta$ | $g$ unchanged; $g_{coh}^L$ becomes $\overline g$ |
| `fold_zero_probability` | distribution | yes | droppable and fixed events; the atom $\bar v$ of a $p_v = 0$ event only if positive-unate in $\tilde g$ | Corollary 5.1; Theorem 10(g) |
| `fold_unit_probability` | distribution | yes | **only fixed events** (and $p = 1$ events whose atoms are positive-unate) | Corollary 5.1; Theorem 10(g); §8.2 |
| eliminate a droppable $p = 1$ event by $g \mapsto \exists_v g$ | distribution | no | yes | Theorem 5; doubles the cone, not proposed as a pass |
| collapse module $M$ with probability $P(M)$ | distribution | yes | only if $P(\omega_L M) = P(M)$ | Proposition 6.1; §8.7 |
| collapse module $M$ with probability $P(\omega_L M)$ | distribution | only if equal | yes, under Proposition 6.1's hypothesis (`atom_split`: both atoms of each event inside or outside $M$) | Proposition 6.1; Theorem 10(h) |

### 7.6 Result reporting

For each hull query a runtime reports:

- the kind (`monotone_hull_probability`), target name, trials $N$, seed;
- the literal mode with $|\Delta|$, the `complement-under-negation` warnings, and `max_initiators_per_cutset`;
- $\hat\theta$ and its interval (method and level; Wilson for $r = 1$ without importance sampling, otherwise a CLT
  interval from the sample variance);
- the per-run bounds $\hat\theta_L \le \hat\theta \le \hat\theta_U$, each with an interval, and the bound rules used
  (for example `lower=unate`, `upper=nnf`);
- the candidate fraction $\hat q$, the number of checked lanes, $r$, and the hit fraction among checked lanes;
- residual statistics: free-event count (median, 95th percentile, maximum), algorithm mix (enumeration, BDD, complete
  search), check time;
- unresolved lanes $u$ (0 for a point estimate, §6.4);
- with importance sampling, the effective sample size.

Projection to PRA entities is as for Boolean results (#181), with the kind in the record.

## 8. Worked examples

All values are exact (rational arithmetic) and were cross-checked by exhaustive enumeration, as were Theorems 1–6 and
8, Proposition 6.1 and the variance formula of §5.3 on random small circuits. The statements of #190 were checked by
exhaustive enumeration on random functions and circuits of at most five events: Theorems 1, 2, 3, 5 and 6, Lemmas 1.1,
1.2 and 3.0 and Corollary 3.1 for all four event classes of §2; Corollary 3.4 and Proposition 7 for droppable and
fixed events; Proposition 6.1 for a droppable module; Theorem 10(b), (c) and (g) on random circuits with direct
complements inside and outside negated gates. The reference tests (`tests/python/test_reference_estimators.py`) check, against
brute force on random projects with complements inside and outside negated gates, the exact value, the cut sets
($K_h$ and $\operatorname{MC}_L$), the per-lane check and the estimator in all three modes (Theorem 10(a), (e), (f)),
and the per-lane check, Theorem 4's bounds and the exact closure of §6.5 for every event class; Theorem 10(g)'s
counterexample is a test. §8.1–§8.7 have no marked literal, so all three modes are the plain hull there.
§8.8–§8.11 are FTP conformance fixtures of #179 (`input/ftp/conformance/`), whose `-EVENT` references the FTP reader
marks as literals.

### 8.1 $F = A \vee BC$, $D = AB$, $p_A = 1$, $p_B = 0.1$, $p_C = 0.2$

- $\operatorname{mcs}(F) = \lbrace A \rbrace, \lbrace B, C \rbrace$; $\operatorname{mcs}(D) = \lbrace A, B \rbrace$;
  neither cut set of $F$ contains $\lbrace A, B \rbrace$, so $\uparrow g = A \vee BC$.
- Boolean: $A = 1$ almost surely, so $g = \neg B$ and $P(g) = 0.9$.
- Hull: $P(A \vee BC) = 1$. $\overline g = g_{coh} = F$, $P = 1$.
- Cut-set quantification of $\operatorname{mcs}(\uparrow g)$: REA $= 1 + 0.02 = 1.02 > 1$; MCUB $= 1$ (exact).
- In the world $A = B = C = 1$, $g = 0$ because $D$ holds, but $\uparrow g = 1$: dropping $A$ leaves $\lbrace B, C \rbrace$,
  dropping $B$ leaves $\lbrace A \rbrace$.

### 8.2 Two-flag delete term: $g = (A \vee B) \wedge \neg(XY)$, $p_A = 0.01$, $p_B = 0.02$, $p_X = p_Y = 1$

- Boolean: $XY = 1$ almost surely, so $P(g) = 0$ exactly.
- Hull: $\operatorname{mcs}(F) = \lbrace A \rbrace, \lbrace B \rbrace$, $\operatorname{mcs}(D) = \lbrace X, Y \rbrace$,
  nothing is deleted: $\uparrow g = A \vee B$, $P = 1 - 0.99 \cdot 0.98 = 0.0298$. $\overline g = g_{coh} = A \vee B$.
- `fold_unit_probability` (setting $X$ and $Y$ to TRUE) gives $g' = 0$ and hull 0: wrong.
- Estimator: with $(g, g_{coh})$ every lane with $A \vee B$ is a candidate ($q = 0.0298$). $X$ and $Y$ are
  negative-unate, so $\underline g = (A \vee B) \wedge \neg(0 \wedge 0) = \uparrow g$ and $q = 0$: no checks.
- The pcfb conformance fixture `delete_term.pcfb` is this circuit with $p_A = 0.25$, $p_B = 0.5$: Boolean 0, hull
  0.625, both exact in binary64.

### 8.3 Exclusion between high-probability events: $F = AC \vee BE$, $DEL = AB$, $p_A = p_B = 0.9$, $p_C = p_E = 0.01$

- No cut set of $F$ contains $\lbrace A, B \rbrace$: $\uparrow g = F = g_{coh}$,
  $P(\uparrow g) = 0.009 + 0.009 - 0.000081 = 0.017919$.
- Boolean: $P(g) = P(AC\neg B) + P(BE\neg A) = 0.0009 + 0.0009 = 0.0018$, ten times smaller. The exclusion holds in
  81% of worlds although it deletes no cut set: the private-model effect in miniature.
- $A$ and $B$ are binate, so $\underline g = g$; $q = P(F \wedge AB) = 0.016119$ and every candidate satisfies the hull
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

Here $\neg B$ stands for a failed-system gate complemented on the success branch (a gate complement), not a direct
complement.

- Boolean: $0.0008 + 0.0002 = 0.001 = P(IA)$; the sequences are disjoint.
- Hull: $\uparrow s_1 = IA$ ($0.001$; the success branch deletes nothing because no cut set of $IA$ contains $B$),
  $\uparrow s_2 = s_2$ ($0.0002$). The class $\uparrow(s_1 \vee s_2) = IA$ has $0.001$, not the sum $0.0012$.

### 8.6 Probability-0 event: $g = (A \vee Z) \wedge \neg(ZB)$, $p_A = 0.1$, $p_Z = 0$, $p_B = 0.5$

$P(\uparrow g) = 0.1$; after folding $Z :=$ FALSE, $\uparrow(A)$ has $0.1$ as well (Corollary 5.1).

### 8.7 Non-monotone module: $M = A\neg B$, $g = M \vee C$, $p_A = 0.5$, $p_B = 0.25$, $p_C = 0.1$

With $\neg B$ a gate complement, $P(\uparrow g) = P(A \vee C) = 0.55$. Collapsing $M$ with $P(M) = 0.375$ gives
$1 - 0.625 \cdot 0.9 = 0.4375$ (wrong); with $P(\uparrow M) = P(A) = 0.5$ it gives $0.55$ (Proposition 6.1).

### 8.8 Direct complement (`process_multi_target`, target `T2`)

`T2 * G-SHARED -B`, `G-SHARED + C D`; $p_B = 0.2$, $p_C = 0.3$, $p_D = 0.4$.

- Atom-split: $\tilde g = (C \vee D) \wedge \bar B$ is monotone over atoms, $K_h = \lbrace \lbrace \bar B, C \rbrace, \lbrace \bar B, D \rbrace \rbrace$:
  the tools' cut sets `{-B, C}`, `{-B, D}`. $P(h) = 0.8 \cdot (1 - 0.7 \cdot 0.6) = 0.464 = P(g)$.
- $\omega_L$ with $L_{T2} = L^+ \cup \lbrace \neg x_B \rbrace$ ($B$ fixed): $g$ is monotone in $C$ and $D$, hence
  $L$-monotone, $\omega_L g = g = h$ (Theorem 10(c)) and $\operatorname{MC}_L(g) = \operatorname{PI}(g) = K_h$.
- Plain hull: $\uparrow g = C \vee D$, cut sets $\lbrace C \rbrace$, $\lbrace D \rbrace$, $P = 0.58$ (the fixture's
  value before #190).
- Cut-set quantification of $\operatorname{MC}_L(g)$: $P(\neg BC) = 0.24$, $P(\neg BD) = 0.32$; REA $= 0.56$,
  MCUB $= 1 - 0.76 \cdot 0.68 = 0.4832 \ge 0.464$ ($B$ occurs only negatively, Proposition 7).
- Estimator: $g = \underline g = \overline g = g_{coh}^L$ (and $\tilde g = \tilde g_{coh}$), no candidate lanes; with
  $L^+$, $g_{coh} = C \vee D$.
- `T1 + G-SHARED A` and `T3 2 A B C` have no direct complement in their cones. `T3` uses $B$ positively; being
  monotone, it would not change even if $B$ were in its $\Delta$.

### 8.9 Direct complement of an unusual name (`names_special`, target `1-TOP`)

`1-TOP + 2A/B--SYS 3--PUMP/C`, `2A/B--SYS * 4E 5F/G -6H--I`, `3--PUMP/C 2 7J/1 8K--2 9L_X`; probabilities 0.1 to 0.6
in that order.

- `6H--I` is split into an atom (fixed under $\omega_L$); $\tilde g$ is monotone over atoms and $h = \omega_L g = g$.
  $K_h = \operatorname{MC}_L(g)$ is $\lbrace 4E, 5F/G, \overline{6H\text{--}I} \rbrace$ and the three pairs of the
  2-of-3 gate, as listed by the tools.
- $P(h) = 1 - (1 - 0.1 \cdot 0.2 \cdot 0.7)(1 - 0.5) = 0.507$; plain hull
  $1 - (1 - 0.02)(1 - 0.5) = 0.51$ (the 2-of-3 gate over 0.4, 0.5, 0.6 has probability 0.5).

### 8.10 The initiator rule (`initiator_rule`)

`TOP + SEQ-1 SEQ-2 SEQ-3 SEQ-4` with `SEQ-1 * IE-1 IE-2 PUMP-A`, `SEQ-2 * IE-1 PUMP-B -SUCC`, `SUCC + PUMP-A VALVE`,
`SEQ-3 * IE-FLAG VALVE -OPER`, `SEQ-4 * IE-FLAG IE-2 PUMP-A`. Initiators $p_{IE\text{-}1} = 0.1$, $p_{IE\text{-}2} = 0.2$,
$p_{IE\text{-}FLAG} = 1$; $p_{PUMP\text{-}A} = 0.3$, $p_{PUMP\text{-}B} = 0.4$, $p_{VALVE} = 0.5$, $p_{OPER} = 0.2$.

- $\Delta = \lbrace OPER \rbrace$; `SUCC` has no event of $\Delta$ in its cone, so Theorem 10(c) holds and the
  atom-split hull equals $\omega_{L_g}$.
- $K_h$: $\lbrace IE\text{-}1, IE\text{-}2, PUMP\text{-}A \rbrace$, $\lbrace IE\text{-}1, PUMP\text{-}B \rbrace$
  (the success branch deletes nothing), $\lbrace IE\text{-}FLAG, VALVE, \overline{OPER} \rbrace$,
  $\lbrace IE\text{-}FLAG, IE\text{-}2, PUMP\text{-}A \rbrace$.
- With $m = 1$, Theorem 10(f) (Corollary 3.4) deletes the first and the last (two initiators each; the
  probability-1 flag counts) and keeps $\lbrace IE\text{-}1, PUMP\text{-}B \rbrace$ and
  $\lbrace IE\text{-}FLAG, VALVE, \overline{OPER} \rbrace$.
- Values (the flag is true almost surely; the remaining cut-set events have disjoint supports):

| | Boolean | atom-split $=\omega_{L_g}$ | plain hull |
|---|---|---|---|
| no rule | 0.45 | $1 - 0.96 \cdot 0.6 \cdot 0.94 = 0.45856$ | $1 - 0.96 \cdot 0.5 \cdot 0.94 = 0.5488$ |
| $m = 1$ | $0.9 \cdot 0.8 \cdot 0.5 \cdot 0.8 = 0.288$ | $1 - 0.96 \cdot 0.6 = 0.424$ | $1 - 0.96 \cdot 0.5 = 0.52$ |

- Boolean with the rule: with `IE-FLAG` true, $\neg\mathrm{atleast}(2; I)$ forces `IE-1` and `IE-2` false, leaving
  $VALVE \wedge \neg OPER$. The literal conjunction removes every world with a second initiator; the hull removes only
  the two cut sets.
- Both tools list exactly the two kept cut sets; with the `I` tags removed, they list the four of $K_h$.

### 8.11 Direct complements under negated gates (`complement_under_negation`)

`TOP + P1 P4`, `P1 + G1 G2`, `G1 * A -G`, `G2 * D -B`, `G + B E`, `P4 * C -H`, `H * C -F`, and the separate root
`P3 + G3 G4`, `G3 * K M`, `G4 * N -M`; $p_A = 0.1$, $p_B = 0.2$, $p_D = 0.3$, $p_E = 0.4$, $p_C = 0.5$, $p_F = 0.6$,
$p_K = 0.25$, $p_M = 0.5$, $p_N = 0.75$.

- `P1` is the first probe of §4.9: $B$ is complemented directly in `G2` and also occurs under the negated gate `G`, so
  the reference tools warn `complement-under-negation`. $h = A \vee D\neg B$, $K_h = \lbrace \lbrace A \rbrace, \lbrace \bar B, D \rbrace \rbrace$;
  $\omega_{L_g} g = \neg B(A \vee D)$; $\uparrow g = A \vee D$.
- `P4` is the second probe: $F$ is complemented directly inside the negated gate `H`. $g = CF$, so
  $\omega_{L_g} g = \uparrow g = g$, but $\tilde g = C \wedge \neg(C \wedge \bar F)$ and $h = C$: above the plain hull
  (Theorem 10(d)).
- `P3` has a direct complement outside any negated gate, so $h = \omega_{L_g} g = g$, but $K_h = \lbrace \lbrace K, M \rbrace, \lbrace \bar M, N \rbrace \rbrace$
  while $\operatorname{MC}_{L_g}(g)$ adds the consensus term $\lbrace K, N \rbrace$.

| target | Boolean | atom-split | $\omega_{L_g}$ | plain hull |
|---|---|---|---|---|
| `P1` | $0.8(1 - 0.94 \cdot 0.7) = 0.2736$ | $0.1 + 0.9 \cdot 0.3 \cdot 0.8 = 0.316$ | $0.8 \cdot 0.37 = 0.296$ | $1 - 0.9 \cdot 0.7 = 0.37$ |
| `P4` | $0.3$ | $0.5$ | $0.3$ | $0.3$ |
| `TOP` | $0.49152$ | $1 - 0.684 \cdot 0.5 = 0.658$ | $0.5072$ | $0.559$ |
| `P3` | $0.5$ | $0.5$ | $0.5$ | $1 - 0.875 \cdot 0.25 = 0.78125$ |

Both vendor tools list exactly $K_h$ for every target; $\omega_{L_g}$ matches neither `P1`, `P4` nor `TOP`, and lists the
extra consensus term on `P3`.

## 9. Open questions

1. Both kinds on one target in the config (§7.1): recommended; confirm in #181.
2. The fuzzybits kind check must land with the pin bump in #183 (§7.3).
3. A writer hint for tighter bounds: deferred until $\underline g$ and $\overline g$ are measured on the private model
   (§7.4).
4. Defaults: $k_{enum}$, BDD and solver budgets, whether to expose subsampling ($r < 1$) and the two-phase variant in
   R1, interval choice for $r < 1$.
5. Whether residual polarity is recomputed per lane (§6.2 step 4): a #178 measurement.
6. Initiator-tagged events in FTP projects (#185): resolved by the rule of §4.8 ($m = 1$ for the vendor tools). Any
   remaining gap between $P(h)$ and cut-set results must be explained by truncation (§4.7) or REA/MCUB. The tools'
   treatment of an initiator with a marked literal was not checked (Theorem 10(f) counts only event atoms).
7. Hull variants of later query kinds (importance measures) are out of scope.
8. Direct complements under a negated gate: resolved on #190 and #199. The atom-split hull is the default
   interpretation of marked literals, which the FTP reader produces (an FTP convention); $\omega_L$ and `negation` are
   explicit overrides; unmarked complements are negations; the lowering warns where `atom_split` and `omega_l` can
   differ.
9. Cut-set listings: $K_h$ has no consensus terms and matches the tools; $\operatorname{MC}_L(g)$ of the `omega_l` option
   can contain consensus terms (§4.2, §8.11). With both atoms of an event in $K_h$ the min-cut upper bound is not an
   upper bound (§4.9), which matters when tool MCUB values are compared with $P(h)$ (#185).
10. The `.pcfb` fields `literal_mode` and `literal_samples` (§7.3) need a pcfb change before 1.2 is tagged.

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
