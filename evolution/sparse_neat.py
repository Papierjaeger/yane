"""Sparse NEAT â€” Lottery Ticket Hypothesis via Iterative Magnitude Pruning.

Finds the "lottery ticket" â€” the minimal sparse sub-network that retains
near-original performance after iterative weight pruning.

Algorithm (IMP):
  1. Evaluate original genome fitness.
  2. Prune the weakest ``prune_frac`` fraction of connections.
  3. Optional: fine-tune via LamarckRefiner.
  4. Re-evaluate.  If fitness â‰¥ original âˆ’ max_fitness_drop, continue.
  5. Repeat until target_sparsity is reached.

Usage::

    from yane.evolution.sparse_neat import find_lottery_ticket, apply_ticket, LotteryTicket

    ticket = find_lottery_ticket(
        genome, fitness_fn,
        target_sparsity=0.5, max_fitness_drop=0.05, iterations=5,
    )
    apply_ticket(genome, ticket)   # disables pruned connections in-place
"""
from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from yane.core.genome import Genome


@dataclass
class LotteryTicket:
    """Result of :func:`find_lottery_ticket`.

    Attributes
    ----------
    mask:
        ``frozenset`` of connection *innovation* IDs that are **kept** active
        in the winning ticket.
    sparsity:
        Fraction of connections that were pruned (0 = none, 1 = all).
    fitness:
        Fitness of the pruned genome (last measured value).
    original_fitness:
        Fitness of the original (unpruned) genome.
    """
    mask: frozenset
    sparsity: float
    fitness: float
    original_fitness: float

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)


def _get_active_connections(genome: "Genome") -> list:
    """Return all enabled connections in the genome."""
    conns = []
    for node in genome.nodes:
        for c in node.connections:
            if c.enabled and c.innovation != -1:
                conns.append(c)
    return conns


def _get_all_connections(genome: "Genome") -> list:
    """Return all connections (enabled + disabled)."""
    conns = []
    for node in genome.nodes:
        for c in node.connections:
            conns.append(c)
    return conns


def find_lottery_ticket(
    genome: "Genome",
    fitness_fn: Callable,
    target_sparsity: float = 0.5,
    max_fitness_drop: float = 0.05,
    iterations: int = 5,
    lamarck_steps: int = 0,
    lamarck_sigma: float = 0.1,
) -> LotteryTicket:
    """Find the sparse lottery ticket via Iterative Magnitude Pruning (IMP).

    Iteratively prunes the weakest connections until ``target_sparsity`` is
    reached or the fitness drop exceeds ``max_fitness_drop``.

    Parameters
    ----------
    genome:
        Trained genome to prune.  Modified in-place during search, then
        restored to original state.
    fitness_fn:
        ``(genome) -> float`` â€” fitness function for evaluation.
    target_sparsity:
        Target fraction of connections to prune (0.0â€“1.0).
    max_fitness_drop:
        Maximum allowed absolute fitness drop from original.
    iterations:
        Number of IMP rounds.  Per round, ``1 - target_sparsity^(1/iterations)``
        fraction of remaining active connections is pruned.
    lamarck_steps:
        If > 0, run this many Lamarckian hill-climbing steps after each prune
        round to compensate for weight removal.
    lamarck_sigma:
        Step size for Lamarckian refinement.

    Returns
    -------
    LotteryTicket
        The best sparse ticket found.
    """
    if not (0.0 <= target_sparsity < 1.0):
        raise ValueError(f"target_sparsity must be in [0, 1), got {target_sparsity}")
    if iterations < 1:
        raise ValueError("iterations must be >= 1")

    # Snapshot original state
    original_states: dict[int, tuple[float, bool]] = {}
    all_conns = _get_all_connections(genome)
    for c in all_conns:
        original_states[id(c)] = (c._weight, c.enabled)

    genome.reset()
    original_fitness = fitness_fn(genome)

    # Per-iteration prune fraction: compound so final sparsity â‰¤ target.
    # We prune from the ORIGINAL total each round so total pruned = target exactly.
    # Each round prunes target_sparsity/iterations of the original count.
    n_original_active = sum(1 for nd in genome.nodes for c in nd.connections
                            if c.enabled and c.innovation != -1)
    # Total connections to prune
    n_total_to_prune = int(n_original_active * target_sparsity)
    # Track how many have been pruned so far
    _n_pruned_so_far = [0]

    best_ticket = LotteryTicket(
        mask=frozenset(c.innovation for c in _get_active_connections(genome)),
        sparsity=0.0,
        fitness=original_fitness,
        original_fitness=original_fitness,
    )

    for iteration in range(iterations):
        active = _get_active_connections(genome)
        if not active:
            break

        # Compute how many to prune in this iteration
        remaining_to_prune = n_total_to_prune - _n_pruned_so_far[0]
        if remaining_to_prune <= 0:
            break  # reached target or target_sparsity=0
        # Spread evenly across remaining iterations
        iters_left = iterations - iteration  # includes current
        n_prune = max(1, round(remaining_to_prune / iters_left))
        n_prune = min(n_prune, remaining_to_prune, len(active))
        if n_prune <= 0:
            break
        # Sort by |weight| ascending â€” weakest first
        by_magnitude = sorted(active, key=lambda c: abs(c._weight))
        to_prune = by_magnitude[:n_prune]

        for c in to_prune:
            c.enabled = False
        _n_pruned_so_far[0] += len(to_prune)
        genome._invalidate_topology()

        # Optional Lamarckian fine-tuning
        if lamarck_steps > 0:
            _lamarck_finetune(genome, fitness_fn, lamarck_steps, lamarck_sigma)

        genome.reset()
        current_fitness = fitness_fn(genome)

        # Remaining active connections
        remaining_active = _get_active_connections(genome)
        n_total_original = sum(
            1 for c in all_conns if original_states[id(c)][1]
        )
        current_sparsity = 1.0 - (len(remaining_active) / max(1, n_total_original))

        if current_fitness >= original_fitness - max_fitness_drop:
            # This ticket is valid
            ticket = LotteryTicket(
                mask=frozenset(c.innovation for c in remaining_active),
                sparsity=current_sparsity,
                fitness=current_fitness,
                original_fitness=original_fitness,
            )
            if ticket.sparsity > best_ticket.sparsity:
                best_ticket = ticket
        else:
            # Fitness drop exceeded â€” stop pruning and restore to last good state
            # Re-enable the connections we just pruned
            for c in to_prune:
                c.enabled = True
            genome._invalidate_topology()
            break

    # Restore genome to original state
    for c in all_conns:
        orig_w, orig_enabled = original_states[id(c)]
        c._weight = orig_w
        if c._weight_arr is not None:
            c._weight_arr[c._weight_idx] = orig_w
        c.enabled = orig_enabled
    genome._invalidate_topology()

    return best_ticket


def _lamarck_finetune(
    genome: "Genome",
    fitness_fn: Callable,
    steps: int,
    sigma: float,
) -> None:
    """Simple hill-climbing fine-tune (avoids LamarckRefiner budget system)."""
    import random
    conns = _get_active_connections(genome)
    nodes = genome.nodes
    genome.reset()
    best_fitness = fitness_fn(genome)
    for _ in range(steps):
        saved_w = [c._weight for c in conns]
        saved_b = [n.bias for n in nodes]
        for c in conns:
            c._weight += random.gauss(0.0, sigma)
            if c._weight_arr is not None:
                c._weight_arr[c._weight_idx] = c._weight
        for n in nodes:
            n.bias += random.gauss(0.0, sigma)
        genome.reset()
        new_fitness = fitness_fn(genome)
        if new_fitness > best_fitness:
            best_fitness = new_fitness
        else:
            for c, w in zip(conns, saved_w):
                c._weight = w
                if c._weight_arr is not None:
                    c._weight_arr[c._weight_idx] = w
            for n, b in zip(nodes, saved_b):
                n.bias = b


def apply_ticket(genome: "Genome", ticket: LotteryTicket) -> None:
    """Disable connections not in the ticket's mask.

    Modifies *genome* in-place.  Connections whose *innovation* is not in
    ``ticket.mask`` are disabled.  Connections with ``innovation == -1``
    (legacy/untracked) are left as-is.

    Parameters
    ----------
    genome:
        Genome to sparsify.
    ticket:
        Result of :func:`find_lottery_ticket`.
    """
    mask = ticket.mask
    for node in genome.nodes:
        for c in node.connections:
            if c.innovation == -1:
                continue  # untracked â€” leave as-is
            if c.innovation not in mask:
                c.enabled = False
    genome._invalidate_topology()

 è  è  V~Ï&é†³Í`Èˆ¸H¿ø evolution/mutation_tracking.py    jò“    jò“       “  «  ¤  è  è  /g±Ôe«Í)òi.*uò6uşh evolution/neuromodulation.py      já»    já»       “  §  ¤  è  è  ±¿ Íg±—kªp'‹`ƒAÓ evolution/online_tuning.py        jÚí    jÚí       “  £  ¤  è  è  NÔ^Şb“TØK@’tÈ9²óÄ@ evolution/onnx_export.py  jÇ´    jÇ´       “  N  ¤  è  è  >Uê´€¶À[îj}R#â\Ñ evolution/operator_scheduler.py   jÕ    jÕ       “  Ÿ  ¤  è  è  8²ÃRÎõª¢ ·°—FmwuJ‡ evolution/output_grouping.py      jvæ    jvæ       “  ”  ¤  è  è  hZ°²AaÍ~üÒLG>BĞ@ evolution/param_registry.py       jÇ´    jÇ´       “  [  ¤  è  è  "¦,Áœá3¼ó4ça72±e|ã evolution/policy.py       jÇ´    jÇ´       “  ’  ¤  è  è CŸlçÍõX?ÄÊo¶€¨«½ evolution/population.py   jvæ    jvæ       “  •  ¤  è  è  2,w”‚GÌÜH/×š¿²¤öÆ evolution/problem_profiler.py     jÇ´    jÇ´       “  E  ¤  è  è  â“XıƒrÂØZ¸êŸ'	Ã evolution/quality_diversity.py    jÇ´    jÇ´       “  W  ¤  è  è  2Ä£Â.\öúÈ:3wÅg	Jéûó©Å evolution/remote_evaluation.py    j%h    j%h       “  ¶  ¤  è  è  5æ1TÊïuÀòâïPŸ1q.˜ evolution/reservoir.py    jÏ    jÏ       “    ¤  è  è  F«ò3ÆhÃì”f‘Ä•TëõÙ-/0 evolution/resource_budget.py      j¥¾    j¥¾       “  p  ¤  è  è  \u µ&³™~13d:Wñ.Üƒñ evolution/safety.py       jÇ´    jÇ´       “  U  ¤  è  è  ó ×òiÏr9ºÇC"2Qœ5ï. evolution/selection_strategy.py   j+    j+       “  °  ¤  è  è  3ò$DÍÔâ¤ærĞÛ‹cí) evolution/self_play.py    jÇ´    jÇ´       “  A  ¤  è  è  %¶¡O¤,1¡¡G>Ğ»Óè|×nÑ evolution/smart_mutation.py       jÇ´    jÇ´       “  V  ¤  è  è  ùIöŸF¹Z3Ë!~Ñ*|ƒ­ evolution/species.py      jğ›    jğ›       “  ª  ¤  è  è  2ÒEôË­eş!ˆá	“’&Äõè4 evolution/stdp.py jÇ´    jÇ´       “  ]  ¤  è  è  ²N‹Ã‚“T ]±ªm<SQ_ evolution/surrogate.py    j-½    j-½       “  ¼  ¤  è  è  ,UÿHm¡›}÷ÏˆR&=pÇj evolution/tflite_export.py        jÇ´    jÇ´       “  ^  ¤  è  è  š©B\î:²Ş‰OÈ¢ƒ“®jµåö evolution/torch_bridge.py jvæ    jvæ       “  ™  ¤  è  è  hk=c¢±Œ(ûÛÂÅïf-s evolution/tracking.py     j@    j@       “  ¬  ¤  è  è  .Y¢§àú‡^F+é1Âjps@óÍi evolution/wasm_export.py  jÇ´    jÇ´       “   h  ¤  è  è  4¶ŸŠ­Ï±ÁøI'û/6Z‰| examples/MNIST/__init__.py        jÇ´    jÇ´       “   g  ¤  è  è   @7ìr'¡’85Kõ[¸Ü®ù examples/MNIST/__main__.py        jÇ´    jÇ´       “   p  ¤  è  è  x[_õnU—.ä½øÈuF›WT“ examples/XOR/__init__.py  jÇ´    jÇ´       “   o  ¤  è  è   @7ìr'¡’85Kõ[¸Ü®ù examples/XOR/__main__.py  jÇ´    jÇ´       “   n  ¤  è  è  :À;Ù…æfìª0Pê[V/ƒƒ examples/XOR/dataset_XOR.json     jÇ´    jÇ´       “     ¤  è  è    æâ›²ÑÖCK‹)®wZØÂäŒS‘ examples/__init__.py      jÇ´    jÇ´       “     ¤  è  è  vLæJR(ş:÷j#ç›€f³¸õ examples/_dataset.py      jÇ´    jÇ´       “   z  ¤  è  è  sÕ³I­gD2¦é×ÅñÔJ˜–ƒë+ )examples/basic_multiplication/__init__.py jÇ´    jÇ´       “   y  ¤  è  è   @7ìr'¡’85Kõ[¸Ü®ù )examples/basic_multiplication/__main__.py jÇ´    jÇ´       “   w  ¤  è  è  k¨4€7Ôè±œé×î±K”´1ïë% /examples/basic_multiplication/create_dataset.py   jÇ´    jÇ´       “   x  ¤  è  è  ´vY¶¾ím+Tn:#KçÜ•'FP 7examples/basic_multiplication/multiplication_table.json   jÇ´    jÇ´       “   ‚  ¤  è  è  ',ÁôL»dòXÊTç1õªhGû #examples/sequence_recall_PI/PI.json       jÇ´    jÇ´       “   †  ¤  è  è  6;G¢Bïön=ÀC[J¤w}L 'examples/sequence_recall_PI/__init__.py   jÇ´    jÇ´       “   …  ¤  è  è   @7ìr'¡’85Kõ[¸Ü®ù 'examples/sequence_recall_PI/__main__.py   jÇ´    jÇ´       “   ƒ  ¤  è  è  F/¾-ŠCŞæó<A™Êá¯ŒÑ -examples/sequence_recall_PI/create_dataset.py     jÇ´    jÇ´       “   „  ¤  è  è 
‡aì%›„_r°9UÈª‘Øp0 +examples/sequence_recall_PI/dataset_PI.json       jÇ´    jÇ´       “     ¤  è  è  w>ÉŸS¥½š¨5Ê™Ò-áêŞ£ *examples/simple_2_2_continuous/__init__.py        jÇ´    jÇ´       “     ¤  è  è   @7ìr'¡’85Kõ[¸Ü®ù *examples/simple_2_2_continuous/__main__.py        jÇ´    jÇ´       “     ¤  è  è  ^;sÚJóñ6ñïY-òˆ	« /examples/simple_2_2_continuous/dataset_2_2.json   jÇ´    jÇ´       “   —  ¤  è  è  ’¡@>3wø£!“¾š½SQU æ2 *examples/simple_3_3_continuous/__init__.py        jÇ´    jÇ´       “   –  ¤  è  è   @7ìr'¡’85Kõ[¸Ü®ù *examples/simple_3_3_continuous/__main__.py        jÇ´    jÇ´       “   •  ¤  è  è  Jİ‚Î÷¥ Ó£'6qç´¦  /examples/simple_3_3_continuous/dataset_3_3.json   jÇ´    jÇ´       “    ¤  è  è    æâ›²ÑÖCK‹)®wZØÂäŒS‘ gui/__init__.py   jÇ´    jÇ´       “  #  ¤  è  è   I<ËÆK¥vËò˜z^½@ıQ¸‚­6 gui/__main__.py   jÇ´    jÇ´       “    ¤  è  è  ä¨q¸iœhì;Ÿ°ô¶¦]GGa gui/_helpers.py   jÇ´    jÇ´       “  2  ¤  è  è  İ¤¦o0Û-åb)«¯c7›şÅ(s gui/_mp_eval.py   jÇ´    jÇ´       “  1  ¤  è  è  Â`¢ø4ìÜªy¯=Ö%æÁŸ gui/canvas.py     jÇ´    jÇ´       “  3  ¤  è  è 2lÄ°¨¢CG­ø™^ø…ßÊ‘(G gui/examples.py   jÉ8    jÉ8       “  5  ¤  è  è  'Ç½oY¸ÇŞY³5ŸnäíõÔ’dÇ gui/interactive_eval.py   jÇ´    jÇ´       “  "  ¤  è  è  ½BCÎw3ƒÌqšLîª‰ÁÅ)xú[ gui/main.py       jÇ´    jÇ´       “    ¤  è  è    æâ›²ÑÖCK‹)®wZØÂäŒS‘ gui/panels/__init__.py    jvæ    jvæ       “    ¤  è  è  ÂÒ0¡dsKc­J–«ØoVáˆ+" gui/panels/left_panel.py  jÇ´    jÇ´       “    ¤  è  è  ÷|èæiôÂTÙ¥™[ƒÅÑ¡äÜ" gui/quality_widgets.py    jÇ´    jÇ´       “    ¤  è  è  Ş–TüŞDç#9†×€ßP‚İ³|T­ gui/remote_config.py      jÇ´    jÇ´       “     ¤  è  è  Ræê,¹úü@Â	õÀifi.Ÿ=P gui/research_features.py  jÇ´    jÇ´       “    ¤  è  è    æâ›²ÑÖCK‹)®wZØÂäŒS‘ gui/tabs/__init__.py      jÇ´    jÇ´       “    ¤  è  è  8Ã_Ö…‰R³è‘Ãè?ß¯r5‡¿’ gui/tabs/aux_tabs.py      jÇ´    jÇ´       “    ¤  è  è  )2òû£]Œ·´Ñ&TC\FÑŠ gui/tabs/comparison_tab.py        jÇ´    jÇ´       “    ¤  è  è  —Ñ¢ÓÁu«¯nÌpôÔûNy®iúnë gui/tabs/inspect_tab.py   jÇ´    jÇ´       “    ¤  è  è  D tté¸·f¯êÉ{`®jbZ£Mİ gui/tabs/runs_tab.py      jvæ    jvæ       “    ¤  è  è ¯5PÓƒi*“qpğk,ùR¥#†ĞÑ gui/tabs/training_tab.py  jÇ´    jÇ´       “    ¤  è  è  Ã²Œrà]‚‡Œ±ÅŸOÇŸ gui/training_sections.py  jÇ´    jÇ´       “  !  ¤  è  è  .I§¿?y¥ÍùŠ£–®¼!õ( gui/window.py     jvæ    jvæ       “  4  ¤  è  è  €şØÜèÜ,&ÉSÁ¨•^`f°~¦ gui/worker.py     j¦#    j¦#       “    ¤  è  è ®æZ'AQEĞñGÌ°NOŸ2D neuro_evolution.py        jÇ´    jÇ´       “    ¤  è  è  *BOíƒ…EÁ&uM@êhÂ42v presets/adaptive_aggressiv.json   jÇ´    jÇ´       “    ¤  è  è  IK)‰x–à†„âÁĞWeØ¸vUİn 'presets/adaptive_analysefreundlich.json   jÇ´    jÇ´       “  ‘  ¤  è  è  ,€_ŠïãRX’ßÙ‹Ø—q   presets/adaptive_balanciert.json  jÇ´    jÇ´       “  ’  ¤  è  è  (P˜ÄdÚ9¨5Ád¡Öõ=W‡È !presets/adaptive_konservativ.json jÇ´    jÇ´       “  “  ¤  è  è  {ù9[Ç!ŠSñ)í­À4)KÉå presets/fast_dataset.json jÇ´    jÇ´       “  ”  ¤  è  è  'q×R“(ûäLÍŒ„w–.»¹Fùƒ $presets/multi_objective_compact.json      jÇ´    jÇ´       “  •  ¤  è  è   ‹Mjş¿#r)x1ÅwKö¢ presets/quality_diversity.json    jÇ´    jÇ´       “  –  ¤  è  è  @)ÖìÄGÿ§åËÉëw‘pÿà“ presets/robust_gym.json   jÇ´    jÇ´       “  œ  ¤  è  è  -ñV½Ù±·—+j:wŸW! 
prompts.md        jÇ´    jÇ´       “     ¤  è  è  »¼Dm¹*Ñ8Ã6È›½è`–jÊ pyproject.toml    jÇ´    jÇ´       “  ™  ¤  è  è   b ™nâÁU\¹+J[‡.<ë;¢€ 
pytest.ini        jÇ´    jÇ´       “     ¤  è  è   ©$”§Xß9–Æ.ó5‰8‡ requirements.txt  jÇ´    jÇ´       “    ¤  è  è   ™İîİ8ˆ¼Ù$
Š„h.¨Á™KÓ run.py    jÇ´    jÇ´       “  '  ¤  è  è    æâ›²ÑÖCK‹)®wZØÂäŒS‘ tests/__init__.py jÇ´    jÇ´       “  $  ¤  è  è  <œÕäÌ<XC³‡‡J”V(³ØrV  tests/fixtures/checkpoint_v1.pkl  jÇ´    jÇ´       “  %  ¤  è  è  4´Y
zÀKDÃ*ŞÕªPÎKşc;  tests/fixtures/checkpoint_v2.pkl  jÇ´    jÇ´       “  &  ¤  è  è  ¤fÄ'PÙ¥* LöİZ; %tests/fixtures/checkpoint_v2.pkl.json     jÇ´    jÇ´       “  7  ¤  è  è  /Ó†juÒ¨<é‘«˜l¨K tests/test_ablation.py    jÇ´    jÇ´       “  -  ¤  è  è  0Vip’HÏ'4'DéšW … !tests/test_activation_function.py jÇ´    jÇ´       “  D  ¤  è  è  i(²Ï*¦[5já¡D[(ç6È tests/test_adaptive_control.py    jÇ´    jÇ´       “    ¤  è  è  3ØïæE&çŞ¤5E0P±á3I\+  !tests/test_adaptive_population.py jÇ´    jÇ´       “  H  ¤  è  è  6+éÛa
õı9
!FÉvè(¼ tests/test_api_configure.py       jÇ´    jÇ´       “  J  ¤  è  è  ¥bÓ^¾“¤¥J¹á¾f¥ÎzmZ 'tests/test_async_descriptors_profile.py   jÉ    jÉ       “  ª  ¤  è  è  +æ	†ºœ±¶`•°Ü]RìıÚF[£: tests/test_attention.py   j}†    j}†       “  œ  ¤  è  è  @®%Şp¸–•®})¤&7ƒ tests/test_augmentation.py        jvæ    jvæ       “  •  ¤  è  è  -NO&Ó$íâ?@ÃbE-ÃÚ6f tests/test_auto_train.py  j¦p    j¦p       “  €  ¤  è  è  ,ÆV¬+ù?9Š9ÑB‘±o´b˜œà] tests/test_bayesian_neat.py       jÇ´    jÇ´       “  C  ¤  è  è  4Æ3S±[	`¦î¯
4ÛRzL úh tests/test_benchmark_gates.py     jÇ´    jÇ´       “  3  ¤  è  è  n]½HvF¡»ıFºùo.OEu%Û tests/test_bugs_and_coverage.py   jÇ´    jÇ´       “  G  ¤  è  è  $>êc²ƒs»dq&®ïÜN±+`Q8 "tests/test_checkpoint_migration.py        jÇ´    jÇ´       “  B  ¤  è  è  ÕhÊps”$ËAo¦°  G’ÛŒBt tests/test_cma_es.py      jÇ´    jÇ´       “  Q  ¤  è  è  Sô¹Î-ãô$ãßÌÇ|.4fÀ tests/test_codec.py       jÇ´    jÇ´       “  ?  ¤  è  è  [‡*FÅÓ+§7^^·x÷‡Y: = tests/test_coevolution.py jÇ´    jÇ´       “  R  ¤  è  è  ¾†`Ÿûk‚à·Ùz(”'ØÙ0 tests/test_config_versioning.py   jÇ´    jÇ´       “  .  ¤  è  è  r†|‚«áñ›ü¹ÜNúêJßÒá¦ tests/test_connection.py  j#s    j#s       “  ±  ¤  è  è  6ÂZ ½+Íàık‘ƒ›Ö÷ì tests/test_continual.py   jØ¬    jØ¬       “  ¡  ¤  è  è  DÏê&¯»ÇÁé´;; *xlî|Ÿ  tests/test_conv_neat.py   j(a    j(a       “  µ  ¤  è  è  1¨¿á½?4Ó(Èiì‡'iÚ tests/test_cooperative.py jÇ´    jÇ´       “  2  ¤  è  è  „Ş¡™oÃŠ©ëı&üX‰şrá &tests/test_crossover_and_speciation.py    jvæ    jvæ       “  ˜  ¤  è  è  æ¥Ç€¼!í…×
´ßLa»²ı tests/test_curiosity.py   jÇµ    jÇµ       “  6  ¤  è  è  +\ämn•ß*D+ûb¡Åúõ´)@ tests/test_curriculum.py  jÇµ    jÇµ       “    ¤  è  è  Q}Í³·¬~À³Ïşà½Øv(å­ tests/test_darts.py       j’    j’       “  °  ¤  è  è  03MXÂ/OWŸk”ş&m‡ô¼ÿ… tests/test_developmental.py       jÇµ    jÇµ       “  L  ¤  è  è  ·`ÃÊh+Df@?ôñ¿(Vo„{0 "tests/test_diagnostics_features.py        jÜ›    jÜ›       “  ¤  ¤  è  è  =ç=áÿ—)UãïóÙ¬mÁ5´š|Š5î tests/test_distillation.py        jÇµ    jÇµ       “  N  ¤  è  è  [‰~e{íÂÔåË45™e _mŒVÌ tests/test_ensemble.py    jÙå    jÙå       “  ¢  ¤  è  è  1Aß]SBm9± åŠ¢Î9q©_ tests/test_es_hyperneat.py        já.    já.       “  ¦  ¤  è  è  ›œ8€/ xäÖÃdd¸'×|´¾ tests/test_experimental.py        jvæ    jvæ       “  ”  ¤  è  è  Y%ñü!f@y>Ë®zq5M}j°æ tests/test_feature_gating.py      jÇµ    jÇµ       “  5  ¤  è  è  %Û±?¼3¡YõÇéõÑÁÙT@Á tests/test_forward_batch.py       jÇµ    jÇµ       “  0  ¤  è  è  bAëzänË¯‚æƒê7ês tests/test_genome.py      j    j       “  ¯  ¤  è  è  .o-ÇŞâé¹uweØâ‡!İTÒfÑ tests/test_grn_encoding.py        jvæ    jvæ       “  —  ¤  è  è  oÖË1q2í‘½:È9æ.ËÒúKùî tests/test_gui_smoke.py   j4    j4       “  ®  ¤  è  è  1µ|ì> ±RšÔ“cüâ¬ÁMW tests/test_h_neat.py      j{m    j{m       “  ›  ¤  è  è  <±³ÅC`¿Ôd\+ù¹›.eb tests/test_hardware_aware.py      jß    jß       “  ¥  ¤  è  è  7Œ9IeÀ”›< GRˆ*IÍÉªZ tests/test_hybrid_neat.py jÇµ    jÇµ       “  U  ¤  è  è  ø]K?XÈşÌ·÷¨õ'
Ì #tests/test_hyperparameter_search.py       jÇµ    jÇµ       “  A  ¤  è  è  
HèÄŸN¿hÙ2RÆğ…°°$ñO tests/test_indirect_encoding.py   jÇµ    jÇµ       “  /  ¤  è  è  "ü4m,alN¬·C=>½†x‡òFÔ‹ tests/test_innovation.py  jÓù    jÓù       “  Ÿ  ¤  è  è  E/ï­¾Oì&\·ôiãş0#*8- tests/test_input_grouping.py      jÉ    jÉ       “    ¤  è  è  N`éP’–€?x·>£Â¹Æú‚ÄrÄ tests/test_interactive_eval.py    jÇµ    jÇµ       “  T  ¤  è  è  )Œÿ\FZ~Ræ³ÜØn_ÉÆ8øƒ tests/test_islands.py     jvæ    jvæ       “  –  ¤  è  è  LâÃ¹æÇ‹ª6—ß©sµÍ«]ö tests/test_knowledge_base.py      jÇµ    jÇµ       “  4  ¤  è  è  -Ü8É¿ÑIdq¶c{ßéïêˆã tests/test_lamarck.py     jÇµ    jÇµ       “  Œ  ¤  è  è  Ô·‰ãá'øñ¢õNeP‘·ø\üW tests/test_lamarck_momentum.py    jÇµ    jÇµ       “  P  ¤  è  è  hãúzÎ†ûˆÊjà¤Ùtèªéa¢ƒ tests/test_landscape.py   jÇµ    jÇµ       “  +  ¤  è  è  .ÕæQ/»2>òÌn´âL)Q~x tests/test_logging.py     j     j        “  «  ¤  è  è  -R+ñ",Q«á“í/ÄïƒfŸŞ tests/test_ltc.py jÇµ    jÇµ       “  F  ¤  è  è  qp´!f R.ºÀ.ğ+C±÷ûÄíG tests/test_matrix_export.py       jÇµ    jÇµ       “  )  ¤  è  è  ¹‚ê ~Œ3Ãê%0Õ§ÏÕ›ÑÃ tests/test_memory.py      jÇµ    jÇµ       “  9  ¤  è  è  7º…‘Õ”ÏHÛÜb“aß,Î¸…aµn tests/test_memory_nodes.py        j$«    j$«       “  ²  ¤  è  è  %RW¾ÖÎw™ğÊ óËÈ‡Lâ tests/test_meta_learning.py       jvæ    jvæ       “  “  ¤  è  è  7§êËö“ò©WĞCé×¬¶˜ôü tests/test_meta_optimizer.py      j'    j'       “  ´  ¤  è  è  .Ö‹QvËIE«¡&›=‡~iü tests/test_minimal_criterion.py   jÇµ    jÇµ       “  @  ¤  è  è  Å±q>hĞŠ«R‚crÔøŞ/C› tests/test_modularity.py  jÇµ    jÇµ       “  =  ¤  è  è  5÷og<×ºÛ«šõ´ÀäİÙX» tests/test_multi_objective.py     jÇµ    jÇµ       “  ;  ¤  è  è  …9ëOí¬z‚ö‘¾ÔÈ4x¸“…Œ tests/test_nes.py jÇµ    jÇµ       “  :  ¤  è  è  œ:dc2Úv_xH\ƒHìÆø†ÔNã tests/test_neuro_evolution.py     jó	    jó	       “  ¨  ¤  è  è  6‘…$U%dYGjW´˜xy+Ls tests/test_neuromodulation.py     jÇµ    jÇµ       “  8  ¤  è  è  /¢ÇniÙÂ³î[6"×Ø*°r¨Q tests/test_normalization.py       jÇµ    jÇµ       “  ,  ¤  è  è  ?lâ¾ËeØ# êŒNqDSR‰ä0" "tests/test_novelty_and_ensemble.py        jÇµ    jÇµ       “  O  ¤  è  è  ñ'`*iæìo.Iiâáïbæ²ÊÔ tests/test_online_tuning.py       jÛ\    jÛ\       “  £  ¤  è  è  35}êy4ÉÿÌa¸ènMldëó±  tests/test_onnx_export.py jÕÑ    jÕÑ       “     ¤  è  è  MPûDß³‰ŸÍ
‰¯á³¾èÏ†L’ tests/test_output_grouping.py     jÇµ    jÇµ       “  K  ¤  è  è  ¦„J~1Î®ûhÑ§=Œ;Çy tests/test_p0_features.py jvæ    jvæ       “  ‘  ¤  è  è  :p÷÷„’pL^ v-–ŠH˜ tests/test_param_registry.py      jÇµ    jÇµ       “  M  ¤  è  è  ÓŞÎ·Q"¶"gIšPæO5ZŞ34 tests/test_plugin_system.py       jÇµ    jÇµ       “  S  ¤  è  è  ÿ`&éeMÎ·ØÑ7ì¤Œóùæ¤‘k tests/test_policy_system.py       jÇµ    jÇµ       “  1  ¤  è  è  1MK‰v‰TY¸_
ÔÓ¹Ÿ®} tests/test_population.py  jÇµ    jÇµ       “    ¤  è  è  ä]KV'^~Nî×Ãv'0á6¥ü #tests/test_post_training_pruning.py       jÇµ    jÇµ       “  E  ¤  è  è  ¦À•íZ4LezĞfR¡à?,úrÚù tests/test_presets.py     jvæ    jvæ       “  ’  ¤  è  è  4—Ğtvëí>LÁ²„$çƒr¬í¤ò tests/test_problem_profiler.py    jÇµ    jÇµ       “  >  ¤  è  è  	8ÓŒ8u¡Á˜ß&VO±Õ,Ö tests/test_quality_diversity.py   jx3    jx3       “  š  ¤  è  è  6-;…ÀvACu#ªÿó½#
åÙ #tests/test_regression_benchmarks.py       jÇµ    jÇµ       “  I  ¤  è  è  $aó ø]Lü³Èˆ÷¯²œF
qõhZ½ tests/test_remote_evaluation.py   j%Ê    j%Ê       “  ³  ¤  è  è  1sÅ®á[ˆDot?€ÉÄY€b§Ú tests/test_reservoir.py   jÏ®    jÏ®       “    ¤  è  è  D æò*ÏsÒI“wSÎoü™0Ó tests/test_resource_budget.py     jÇµ    jÇµ       “  (  ¤  è  è  ŞáÑ*s‘æ1>wUgzC•+ tests/test_rss_stability.py       jÇµ    jÇµ       “  <  ¤  è  è  6)²á7¥ßi—V:è`aÿ£ü tests/test_sa.py  j¦¡    j¦¡       “  ‚  ¤  è  è  7ùşøí®ÿGn§Ù|JÃc»ÿx î tests/test_safety_neat.py jŞ    jŞ       “  ­  ¤  è  è  5æ‚4b”+³qÁ‘xÈÇÌ;ğÏ† tests/test_self_play.py   jÇµ    jÇµ       “    ¤  è  è  "Åevò’ëõn•ñ!0§ì©g*Ë tests/test_shared_weights.py      jÇµ    jÇµ       “  *  ¤  è  è  JC/İV¼RSJ)â;Y	¦¾ÿi‚§ tests/test_smart_mutation.py      j¦È    j¦È       “  …  ¤  è  è  *úvGÃ{^ì=HÃ¢!•ç´úÆW tests/test_sparse_neat.py jò     jò        “  §  ¤  è  è  <&›×+D¤ÙÃBvLşøá$KÀ° tests/test_stdp.py        jÇµ    jÇµ       “  V  ¤  è  è  
S>Ù±vöÊ”›*PÓëæp€3ŠÕ tests/test_surrogate.py   j§"    j§"       “  ‹  ¤  è  è  ;	Õ.MÊL]Mböm1SªÖ8YFs !tests/test_symbolic_regression.py jR    jR       “  ¬  ¤  è  è  1§W­Ä¬>ò8Ë	ÿ GT °ˆ‚¹Ñ !tests/test_temporal_speciation.py j¦é    j¦é       “  ˆ  ¤  è  è  'Q%­ºèZÜ3 '%Òº[s tests/test_tflite_export.py       jÇµ    jÇµ       “  W  ¤  è  è  Áºa8vgV‚­Ì	  “‡8¡3Ó tests/test_torch_bridge.py        jvæ    jvæ       “  ™  ¤  è  è  ,e*•ä¸¤«=xj†>†mK tests/test_tracking.py    jJ    jJ       “  ©  ¤  è  è  5É?Â ·å†7¸p’*	İl¨h#½
 tests/test_wasm_export.py jÇµ    jÇµ       “  È  ¤  è  è    æâ›²ÑÖCK‹)®wZØÂäŒS‘ util/__init__.py  jÇµ    jÇµ       “  Ú  ¤  è  è  (ÜÖÀVKÿa*İ¶®Ï!&®× util/activation.py        jÇµ    jÇµ       “  Ì  ¤  è  è  "S“#ãG†ıa—š×Ó¥Ú2×~œó util/logger.py    jÇµ    jÇµ       “  Ê  ¤  è  è  ;ıoT+`èı
zµ‹Vµ * util/normalization.py     jÇµ    jÇµ       “  Ë  ¤  è  è  	9 ‡ıëÙñvÅÕÇÔºFÓóIªÀ,< util/presets.py   jÇµ    jÇµ       “  Ï  ¤  è  è  Ch¸û†géMz‘õv½sê•ò=Á util/report.py    jÇµ    jÇµ       “  É  ¤  è  è  @¹eRFEiàïqIÓ€òãê util/resource_guard.py    jvæ    jvæ       “  Û  ¤  è  è  0´âÊ§
éŠ^ <ax±…Ì4ÉW util/run_database.py      jÇµ    jÇµ       “  Í  ¤  è  è  ‘ eÊ"k,Ã§ËŸ7»ˆé'ÃC~¦ util/run_history.py       jÇµ    jÇµ       “  Î  ¤  è  è  õ-ùĞË¶Ûr|ÊÀ7¡‰Õ°cãKY util/run_report.py        TREE  r -1 10
api 7 1
Ry½ƒÙfÑ}g†<Å{;UˆŸh0routes 3 0
‹úeë­àw[[J·Ygui 22 2
\Ÿá~ŒDNçñ<:gÚÄúk¡ tabs 6 0
¶|ı²fÓI*Ø6Ñ¯çŒ«panels 2 0
~ˆy”XJeúÂnƒm ¤core -1 0
util 10 0
é÷)qÑÒ»ËZ²Ÿtweıtests -1 1
fixtures 3 0
ÜúG;SÌò’äCÿÂÉ¶xA.İ.github 1 0
XÂü5¤¢4\rTÍÂß¢qÛ©presets 8 0
}4ĞÛVóàÆŒÏğ›ÑlE¼~EH˜examples 22 6
£Êà¸'<Ÿ£Wê›I¥êøæ¾XOR 3 0
0?PŞ‰Ï>H33úò¨‰¹¸MNIST 2 0
ù"KI‹®×ïĞ² ‚Â'İÒRúsequence_recall_PI 5 0
oê?”b¼„§ÎÃ[XŒƒô,basic_multiplication 4 0
´96„ª®’3RÜß'¿5k,4fsimple_2_2_continuous 3 0
¥Û~ìî|iAT†>jh-'qcÜsimple_3_3_continuous 3 0
…ùX'zy<àt[£gU×NA†“evolution -1 0
benchmarks 24 0
ı
ûÌE¸ [;ĞÇº@İ1â^¶ğsË˜ ³´ÜusmËTÏ›Ó‡