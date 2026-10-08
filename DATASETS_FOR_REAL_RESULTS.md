# Real Datasets to Move FedRL-CS Off Synthetic Data

Goal: replace the synthetic task AND the modeled energy/network with real,
citable data, so the paper's biggest reviewer risk goes away. Below are the
options, ranked by how well they fit *this* project (hierarchical client selection
with energy + fairness). Each entry says what it fixes, how it maps to our code,
and how to get it.

Our two weaknesses a dataset must address:
(1) the classification task is synthetic; (2) the client system state
(compute, battery, bandwidth, availability) is modeled, so "energy" is a proxy.

=====================================================================
1. FedScale  —  TOP PICK (fixes BOTH weaknesses at once)
=====================================================================
FedScale is the benchmark built by the same lab as Oort, specifically for
realistic federated learning and client selection.

What it gives us:
- Real task data with natural per-user partitions: FEMNIST (handwritten
  characters by ~3.5k writers), Google Speech, OpenImage, Reddit, StackOverflow,
  and more. Client = a real user, so the non-IID split is real, not Dirichlet.
- Real CLIENT SYSTEM TRACES — this is the key part for us:
    * device compute capability (from the AIBench device measurements),
    * network throughput (from the MobiPerf bandwidth dataset),
    * client availability / duration (from the FLASH user-behavior trace,
      ~136k users, ~180M battery/screen events).
  These plug straight into our client-attributes module and make energy,
  bandwidth, stragglers, and churn REAL measured quantities instead of draws.
Why it is the best fit: it removes both weaknesses in one move, and because Oort
was evaluated on it, our client-selection comparison lands on the field's
standard testbed.
How to get it: GitHub SymbioticLab/FedScale; install and run
`fedscale dataset download <name>` (FEMNIST is the natural first target).
Bring me: the FEMNIST partition + the device/network/availability traces.

=====================================================================
2. LEAF  —  EASIEST classic real-partition benchmark (fixes weakness 1)
=====================================================================
The original federated benchmark. Real, naturally-partitioned datasets:
- FEMNIST: 62-class handwritten characters, partitioned by writer (~3.5k clients).
- Shakespeare: next-character prediction, partitioned by speaking role.
- Sent140: tweet sentiment, partitioned by Twitter user.
- Also CelebA, Reddit.
What it fixes: the data becomes real with genuine client heterogeneity. It does
NOT give system/energy traces, so we would still model energy (pair it with
FedScale traces if you want both). Lightest to set up and very widely cited.
How to get it: GitHub TalwalkarLab/leaf; run the dataset generation script (e.g.
FEMNIST) to produce per-client train/test JSON.
Bring me: the generated per-client FEMNIST (or Shakespeare) data.

=====================================================================
3. FLamby  —  REAL cross-silo (fits the "hospitals / banks can't share" story)
=====================================================================
Seven real cross-silo datasets where each client is a real institution:
- Fed-ISIC2019 (skin-lesion images across 6 hospitals), Fed-Heart-Disease
  (4 sites), Fed-IXI (brain MRI), Fed-TCGA-BRCA, Fed-Camelyon16, Fed-KiTS19,
  Fed-LIDC-IDRI.
What it fixes: real data with a real, natural client = institution partition, and
a strong real-world motivation (privacy law makes pooling illegal). This is the
cross-silo regime (few, large clients) rather than cross-device (many small
clients), so it exercises a different and complementary setting — a good second
experiment, and it maps well onto our "cluster = site" framing.
How to get it: GitHub owkin/FLamby (some datasets need a data-use agreement,
e.g. the medical ones — check each dataset's terms).
Bring me: whichever FLamby dataset clears its license (Fed-Heart-Disease and
Fed-ISIC2019 are the easiest starts).

=====================================================================
4. TensorFlow Federated (TFF) built-in datasets  —  convenient
=====================================================================
Federated EMNIST, Shakespeare, StackOverflow, and a federated CIFAR-100, all
pre-partitioned by user. Trivial to load if you use TensorFlow; a good quick path
to a real-data result. Same limitation as LEAF: no system/energy traces.

=====================================================================
5. Domain-specific (your fintech angle)  —  optional, non-standard
=====================================================================
Given the lending/fraud context, a real tabular credit-risk or fraud dataset
partitioned by institution (each bank/lender = a client) would make the
motivation land hard for a fintech audience. There is no standard FL-partitioned
fintech benchmark, so this would be a self-constructed cross-silo split of a
public credit/fraud dataset. Treat it as an extra, not the primary, experiment,
and keep FedScale/LEAF as the reproducible standard the paper is judged on.

=====================================================================
RECOMMENDATION
=====================================================================
Primary: FedScale with FEMNIST + its device/network/availability traces. It is
the one option that removes BOTH the synthetic-data and the modeled-energy
criticisms, and it is the standard testbed for client selection.
Fast fallback: LEAF FEMNIST for the data, keep our energy model but cite it as a
model (weaker, but a real-data result quickly).
Second experiment for breadth: one FLamby cross-silo dataset.

=====================================================================
WHAT I NEED FROM YOU / WHAT I WILL DO
=====================================================================
When you bring the data, the code change is contained:
- swap our synthetic `generate_dataset()` for a per-client real-data loader
  (client = writer / user / site); the RL selection, policy federation, metrics,
  and all analyses stay as they are;
- if FedScale traces are included, replace the drawn compute/bandwidth/battery in
  the client-attributes module with sampled real trace values, which makes energy
  and stragglers measured rather than modeled;
- assign the real clients to edge clusters (by device tier, region if available,
  or a fixed random grouping) to keep our two-tier structure;
- then re-run the exact same pipeline (main comparison, significance, ablations,
  MAPPO, scale) and regenerate the tables and figures.
Bring the dataset folder (or tell me which one and I will fetch the loader), and
I will wire it in and produce the real-data results.
