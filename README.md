# Verified Federated Learning with DP-SGD, zkVM Proofs, and On-Chain Privacy Enforcement

**Prototype demo for the Ethereum Foundation PhD Fellowship 2026**

A Rust implementation demonstrating privacy-preserving multi-institutional AI training for urban building maintenance in Hong Kong. The system combines vertical federated learning, differentially private SGD, zero-knowledge virtual machine proofs, and Ethereum Layer 2 smart contract enforcement into a single auditable pipeline.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────────┐
│ Participant Layer│     │   Proof Layer    │     │  On-Chain Layer (L2) │
│ (Vertical Split) │     │                  │     │                      │
│                  │     │  ┌────────────┐  │     │  ┌────────────────┐  │
│ Buildings Dept ──┼──e1─┼─>│  zkVM      │  │     │  │ Verifier       │  │
│ Property Mgr  ──┼──e2─┼─>│  Prover    │  │     │  │ Contract       │  │
│ Inspection    ──┼──e3─┼─>│  (SP1/R0)  │  │     │  └───────┬────────┘  │
│ Owners' Corp  ──┼──e4─┼─>│            │  │     │          │           │
│                  │     │  └─────┬──────┘  │     │  ┌───────▼────────┐  │
│                  │     │        │         │     │  │ State Root     │  │
│                  │     │  ┌─────▼──────┐  │     │  │ (Merkle)       │  │
│                  │     │  │  DP-SGD    │  │     │  └───────┬────────┘  │
│                  │     │  │ Clip+Noise │──┼─eps─┼─>│       │           │
│                  │     │  └─────┬──────┘  │     │  ┌───────▼────────┐  │
│                  │     │        │         │     │  │ Privacy Budget │  │
│                  │     │  ┌─────▼──────┐  │     │  │ eps_total cap  │  │
│                  │     │  │ IVC Agg.   │──┼─pi──┼─>└───────┬────────┘  │
│                  │     │  └─────┬──────┘  │     │          │           │
│                  │     │        │         │     │  ┌───────▼────────┐  │
│                  │◄────┼──grad──┤         │     │  │ Audit Log      │  │
│                  │     │  ┌─────▼──────┐  │     │  │ Per-round      │  │
│                  │     │  │Coordinator │  │     │  └────────────────┘  │
│                  │     │  │ Top model  │  │     │                      │
│                  │     │  └────────────┘  │     │                      │
└─────────────────┘     └──────────────────┘     └──────────────────────┘
                                                           │
                                              ┌────────────▼────────────┐
                                              │    Model Output (MBIS)  │
                                              │ Demand Predictor        │
                                              │ Cost Estimator          │
                                              └─────────────────────────┘
```

## How It Works

**Vertical federated learning** addresses the scenario where multiple institutions (Buildings Department, property managers, inspection bodies, owners' corporations) each hold different attribute columns about the same set of buildings but cannot share raw data. Each participant trains a local subnetwork over its own feature partition and sends only intermediate embeddings to the coordinator.

**DP-SGD** (differentially private stochastic gradient descent) ensures formal (epsilon, delta)-privacy guarantees. Each participant clips per-sample gradients to a declared sensitivity bound and injects calibrated Gaussian noise at each update. Cumulative privacy loss is tracked via Renyi differential privacy composition.

**zkVM proofs** provide cryptographic verification that each participant actually executed the DP-SGD protocol correctly. Every training step runs inside a zero-knowledge virtual machine, producing a succinct proof that the declared privacy parameters were enforced. No participant can claim compliance while secretly omitting noise or weakening clipping.

**On-chain enforcement** aggregates per-participant proofs via IVC (incrementally verifiable computation) into a single proof per round, which is verified by a smart contract on Ethereum L2. The contract maintains the model's Merkle root, tracks cumulative epsilon, and permanently halts training when the pre-agreed privacy budget is exhausted.

## Project Structure

```
verified-vfl-demo/
├── Cargo.toml                 # Rust project configuration
├── LICENSE                    # MIT License
├── README.md
├── src/
│   ├── main.rs                # CLI entry point: runs full pipeline
│   ├── lib.rs                 # Module declarations
│   ├── types.rs               # Core data types and structures
│   ├── dp_sgd.rs              # DP-SGD: clipping, noise, Renyi accounting
│   ├── zkvm_mock.rs           # Mock zkVM proof generation/verification
│   ├── ivc.rs                 # IVC proof aggregation
│   ├── participant.rs         # VFL participant with local subnetwork
│   ├── coordinator.rs         # Top model coordinator
│   ├── onchain.rs             # Mock on-chain contract logic
│   └── privacy_budget.rs      # Privacy budget tracker with hard cap
├── web/
│   └── index.html             # Interactive web demo (open in browser)
└── data/
    └── sample_buildings.json  # Sample Hong Kong building data
```

## Quick Start

### Prerequisites

- [Rust](https://rustup.rs/) (1.70+)

### Run the CLI Demo

```bash
git clone https://github.com/user/verified-vfl-demo.git
cd verified-vfl-demo
cargo run
```

This executes the full pipeline: initializes 4 institutional participants with vertically partitioned building data, runs multiple training rounds with DP-SGD inside mock zkVM provers, aggregates proofs via IVC, submits them to the mock on-chain contract, and demonstrates privacy budget enforcement (training halts automatically when the budget ceiling is reached).

### Run Tests

```bash
cargo test
```

### Interactive Web Demo

Open `web/index.html` in any browser. The demo lets you adjust privacy parameters (epsilon per round, budget ceiling, clipping bound, number of participants) and observe the training loop with live architecture animation, loss/budget charts, and an on-chain audit log.

## Module Overview

| Module | Description |
|--------|-------------|
| `types.rs` | Shared types: `BuildingRecord`, `Embedding`, `Gradient`, `ZkProof`, `AggregatedProof`, `OnChainState`, prediction outputs |
| `dp_sgd.rs` | Gradient clipping to bound C, calibrated Gaussian noise generation from committed seeds, weight update, Renyi DP accountant |
| `zkvm_mock.rs` | Simulated zkVM proof generation (SHA-256 commitments) and verification; seed commitment via mock Poseidon hash |
| `ivc.rs` | IVC aggregation of per-participant proofs into a single round proof; hash-chain verification |
| `participant.rs` | VFL participant: local subnetwork (linear + ReLU), forward pass producing embeddings, backward pass with DP-SGD + proof generation |
| `coordinator.rs` | Top model: concatenates embeddings, forward pass (linear + sigmoid), loss computation, gradient splitting back to participants |
| `onchain.rs` | Mock Ethereum L2 contract: proof verification, Merkle root update, epsilon tracking, budget enforcement, audit log |
| `privacy_budget.rs` | Privacy budget with hard cap enforcement, utilization tracking |

## Cryptographic Simplifications

This prototype uses SHA-256 hash commitments to simulate zkVM proofs. In a production deployment:

- **zkVM**: Each DP-SGD step runs inside RISC Zero or SP1, producing STARK/SNARK proofs of 3-4 KB with 2-5 ms verification time
- **Seed commitment**: Uses Poseidon hash (efficient inside zkVM arithmetic circuits) instead of SHA-256
- **IVC aggregation**: Uses Nova/SuperNova folding schemes instead of hash chaining
- **On-chain verification**: Solidity verifier contract with < 300K gas per round using precompiled elliptic curve operations

## Context

This demo accompanies an Ethereum Foundation PhD Fellowship 2026 application under the "Coordination Rails for City Governments & Urban Systems" RFP. The full research project integrates real building data from Hong Kong's Mandatory Building Inspection Scheme with production zkVM infrastructure and Ethereum L2 deployment.

## Authors

- **Xu Pei** — Department of Architecture and Civil Engineering, City University of Hong Kong
- **Kurt Pan** — The Hong Kong Polytechnic University

## License

MIT
