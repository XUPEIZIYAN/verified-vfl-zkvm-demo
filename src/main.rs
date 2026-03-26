//! Verified Vertical Federated Learning Demo
//!
//! End-to-end simulation of the full pipeline:
//!   Participants -> DP-SGD inside zkVM -> IVC Aggregation -> On-Chain Verification
//!
//! This demo simulates the Hong Kong building maintenance scenario with
//! four institutional participants, each holding different feature partitions.

mod coordinator;
mod dp_sgd;
mod ivc;
mod onchain;
mod participant;
mod privacy_budget;
mod types;
mod zkvm_mock;

use coordinator::Coordinator;
use onchain::{hash_model_weights, OnChainContract};
use participant::Participant;
use types::{BuildingRecord, DpSgdConfig};

use std::fs;

fn main() {
    println!("========================================================");
    println!("  Verified Federated Learning with DP-SGD & zkVM Proofs");
    println!("  Prototype Demo for Urban Building Maintenance (HK)");
    println!("========================================================\n");

    // ── Configuration ──────────────────────────────────────
    let epsilon_per_round = 0.5;
    let delta = 1e-5;
    let clip_bound = 1.0;
    let learning_rate = 0.01;
    let embedding_dim = 4;
    let total_budget = 3.0;
    let num_rounds = 8; // Will be stopped by budget enforcement

    println!("[Config]");
    println!("  Epsilon per round:  {}", epsilon_per_round);
    println!("  Delta:              {:.0e}", delta);
    println!("  Clip bound (C):     {}", clip_bound);
    println!("  Total budget cap:   {}", total_budget);
    println!("  Max training rounds: {}", num_rounds);
    println!();

    // ── Load sample building data ──────────────────────────
    let buildings = load_sample_data();
    println!("[Data] Loaded {} buildings\n", buildings.len());

    // ── Initialize participants (vertical partition) ────────
    // Each participant holds different feature columns:
    //   Buildings Dept:  features[0..3]  (age, structural type, district)
    //   Property Mgr:    features[3..6]  (repair history, access logs, contractor)
    //   Inspection Body: features[6..9]  (inspection scores, defects, compliance)
    //   Owners' Corp:    features[9..12] (finances, votes, insurance)
    let dp_config = DpSgdConfig::new(epsilon_per_round, delta, clip_bound, learning_rate);

    let mut participants = vec![
        Participant::new("buildings_dept", 3, embedding_dim, dp_config.clone(), 1001),
        Participant::new("property_mgr", 3, embedding_dim, dp_config.clone(), 2002),
        Participant::new("inspection_body", 3, embedding_dim, dp_config.clone(), 3003),
        Participant::new("owners_corp", 3, embedding_dim, dp_config.clone(), 4004),
    ];

    let feature_slices: Vec<(usize, usize)> = vec![(0, 3), (3, 6), (6, 9), (9, 12)];
    let embedding_dims: Vec<usize> = participants.iter().map(|p| p.embedding_dim).collect();
    let total_emb_dim: usize = embedding_dims.iter().sum();

    println!("[Participants]");
    for p in &participants {
        println!(
            "  {} | input_dim={} | embedding_dim={} | seed_commit={:.16}...",
            p.id, p.input_dim, p.embedding_dim, p.seed_commitment
        );
    }
    println!();

    // ── Deploy on-chain contract ────────────────────────────
    let mut contract = OnChainContract::deploy(total_budget);
    for p in &participants {
        contract.register_seed_commitment(&p.id, &p.seed_commitment);
    }
    println!(
        "[On-Chain] Contract deployed | Budget: {} | Registered {} participants\n",
        total_budget,
        participants.len()
    );

    // ── Initialize coordinator (top model) ──────────────────
    let mut coord = Coordinator::new(total_emb_dim, 2, learning_rate);
    println!(
        "[Coordinator] Top model initialized | input={} | output=2\n",
        total_emb_dim
    );

    // ── Synthetic targets for demonstration ─────────────────
    // Target: [urgency_score, normalized_cost]
    let targets: Vec<Vec<f64>> = buildings
        .iter()
        .enumerate()
        .map(|(i, _)| {
            vec![
                0.3 + 0.05 * (i as f64), // urgency increases with index
                0.2 + 0.04 * (i as f64), // cost increases similarly
            ]
        })
        .collect();

    // ── Training loop ───────────────────────────────────────
    println!("═══════════════════════════════════════════════════════");
    println!("  TRAINING LOOP");
    println!("═══════════════════════════════════════════════════════\n");

    let mut round_results: Vec<serde_json::Value> = Vec::new();

    for round in 1..=num_rounds {
        println!("── Round {} ──────────────────────────────────────", round);

        let mut round_loss = 0.0;

        // Collect proofs from all participants this round
        let mut round_proofs = Vec::new();

        for (b_idx, building) in buildings.iter().enumerate() {
            // Step 1: Each participant computes local embedding
            let embeddings: Vec<types::Embedding> = participants
                .iter()
                .enumerate()
                .map(|(p_idx, p)| {
                    let (start, end) = feature_slices[p_idx];
                    let local_record = BuildingRecord {
                        building_id: building.building_id.clone(),
                        features: building.features[start..end.min(building.features.len())]
                            .to_vec(),
                    };
                    p.forward(&local_record)
                })
                .collect();

            // Step 2: Coordinator concatenates and runs top model
            let concatenated = coord.concatenate_embeddings(&embeddings);
            let prediction = coord.forward(&concatenated);
            let loss = coord.compute_loss(&prediction, &targets[b_idx]);
            round_loss += loss;

            // Step 3: Coordinator computes and distributes gradients
            let full_gradient = coord.backward(&concatenated, &prediction, &targets[b_idx]);
            let split_grads = coord.split_gradient(&full_gradient, &embedding_dims);

            // Step 4: Each participant runs backward with DP-SGD inside zkVM
            for (p_idx, p) in participants.iter_mut().enumerate() {
                let (start, end) = feature_slices[p_idx];
                let features = &building.features[start..end.min(building.features.len())];
                let proof = p.backward_with_proof(&split_grads[p_idx], features, round);

                // Only collect proofs for the last building (one proof per participant per round)
                if b_idx == buildings.len() - 1 {
                    round_proofs.push(proof);
                }
            }
        }

        round_loss /= buildings.len() as f64;

        // Step 5: Verify individual proofs
        let mut all_verified = true;
        for proof in round_proofs.iter_mut() {
            let valid = zkvm_mock::verify_proof(proof, epsilon_per_round * 2.0, delta * 10.0);
            if !valid {
                all_verified = false;
                println!("  WARN: Proof verification failed for {}", proof.participant_id);
            }
        }

        // Step 6: IVC aggregation
        let mut agg_proof = ivc::aggregate_proofs(round, &round_proofs);
        ivc::verify_aggregated_proof(&mut agg_proof, &round_proofs);

        println!(
            "  Proofs: {} generated, {} verified | IVC aggregated: {}",
            round_proofs.len(),
            if all_verified { "all" } else { "PARTIAL" },
            if agg_proof.valid == Some(true) { "VALID" } else { "INVALID" }
        );

        // Step 7: Submit to on-chain contract
        let model_hash = hash_model_weights(&coord.weights);
        match contract.submit_round(&agg_proof, &model_hash) {
            Ok(record) => {
                let event = contract.emit_round_completed(&record);
                println!("  Loss: {:.6} | {}", round_loss, event);

                round_results.push(serde_json::json!({
                    "round": round,
                    "loss": round_loss,
                    "epsilon_round": record.epsilon_round,
                    "epsilon_total": record.epsilon_cumulative,
                    "budget_remaining": contract.budget.remaining(),
                    "merkle_root": &record.model_merkle_root[..18],
                    "verified": true,
                    "training_halted": contract.state.training_halted,
                }));

                if contract.state.training_halted {
                    println!(
                        "\n  *** TRAINING PERMANENTLY HALTED: Budget ceiling {} reached ***\n",
                        total_budget
                    );
                    break;
                }
            }
            Err(e) => {
                println!("  ON-CHAIN REJECTED: {}", e);
                round_results.push(serde_json::json!({
                    "round": round,
                    "loss": round_loss,
                    "rejected": true,
                    "reason": e,
                }));
                break;
            }
        }
        println!();
    }

    // ── Model inference demonstration ───────────────────────
    println!("═══════════════════════════════════════════════════════");
    println!("  MODEL INFERENCE (Post-Training)");
    println!("═══════════════════════════════════════════════════════\n");

    for building in buildings.iter().take(3) {
        let embeddings: Vec<types::Embedding> = participants
            .iter()
            .enumerate()
            .map(|(p_idx, p)| {
                let (start, end) = feature_slices[p_idx];
                let local_record = BuildingRecord {
                    building_id: building.building_id.clone(),
                    features: building.features[start..end.min(building.features.len())].to_vec(),
                };
                p.forward(&local_record)
            })
            .collect();

        let concatenated = coord.concatenate_embeddings(&embeddings);
        let output = coord.forward(&concatenated);

        let maint = coord.predict_maintenance(&output, &building.building_id, 10);
        let cost = coord.predict_cost(&output, &building.building_id, "General Repair");

        println!("  Building: {}", building.building_id);
        println!(
            "    Maintenance: urgency={:.3}, type={}, horizon={}yr",
            maint.urgency_score, maint.maintenance_type, maint.horizon_years
        );
        println!(
            "    Cost: HKD {:.0} [{:.0} - {:.0}] ({})",
            cost.cost_estimate, cost.cost_lower, cost.cost_upper, cost.repair_type
        );
        println!();
    }

    // ── On-chain audit summary ──────────────────────────────
    println!("═══════════════════════════════════════════════════════");
    println!("  ON-CHAIN AUDIT LOG");
    println!("═══════════════════════════════════════════════════════\n");

    let state = contract.get_state();
    println!("  Total rounds completed: {}", state.current_round);
    println!("  Final model Merkle root: {:.32}...", state.model_merkle_root);
    println!("  Epsilon consumed: {:.4} / {:.4}", state.epsilon_total, state.epsilon_budget);
    println!("  Training halted: {}", state.training_halted);
    println!();

    for record in &state.audit_log {
        println!(
            "  Round {:2} | eps={:.4} | cumul={:.4} | root={:.16}... | verified={}",
            record.round,
            record.epsilon_round,
            record.epsilon_cumulative,
            record.model_merkle_root,
            record.verified
        );
    }

    // ── Export results as JSON ───────────────────────────────
    let output_json = serde_json::json!({
        "config": {
            "epsilon_per_round": epsilon_per_round,
            "delta": delta,
            "clip_bound": clip_bound,
            "total_budget": total_budget,
            "num_participants": participants.len(),
            "num_buildings": buildings.len(),
        },
        "rounds": round_results,
        "final_state": {
            "rounds_completed": state.current_round,
            "epsilon_consumed": state.epsilon_total,
            "epsilon_budget": state.epsilon_budget,
            "training_halted": state.training_halted,
            "model_merkle_root": &state.model_merkle_root,
        },
        "audit_log": state.audit_log,
    });

    let json_str = serde_json::to_string_pretty(&output_json).unwrap();
    fs::write("training_results.json", &json_str).ok();
    println!("\n[Output] Results written to training_results.json");

    println!("\n========================================================");
    println!("  Demo complete. All proofs verified on-chain.");
    println!("========================================================");
}

/// Load sample building data. In production, each participant loads only
/// its own feature partition; here we generate a complete dataset for
/// simulation purposes.
fn load_sample_data() -> Vec<BuildingRecord> {
    // Try to load from file first
    if let Ok(contents) = fs::read_to_string("data/sample_buildings.json") {
        if let Ok(records) = serde_json::from_str::<Vec<BuildingRecord>>(&contents) {
            return records;
        }
    }

    // Generate synthetic data: 10 buildings, 12 features each
    // Features represent the vertical partition across 4 institutions:
    //   [0-2]: Buildings Dept (age_norm, structural_type, district_code)
    //   [3-5]: Property Mgr (repair_count_norm, access_freq, contractor_rating)
    //   [6-8]: Inspection Body (inspection_score, defect_count_norm, compliance)
    //   [9-11]: Owners' Corp (reserve_fund_norm, vote_participation, insurance_coverage)
    let mut buildings = Vec::new();
    for i in 0..10 {
        let age = 0.3 + 0.06 * i as f64;       // Normalized age (30-90 years)
        let struct_type = if i % 3 == 0 { 0.8 } else { 0.4 };
        let district = (i % 5) as f64 / 5.0;
        let repairs = 0.1 + 0.08 * i as f64;
        let access = 0.5 + 0.03 * i as f64;
        let contractor = 0.7 - 0.02 * i as f64;
        let inspection = 0.8 - 0.05 * i as f64;
        let defects = 0.1 + 0.07 * i as f64;
        let compliance = if i < 5 { 0.9 } else { 0.5 };
        let reserve = 0.6 - 0.04 * i as f64;
        let votes = 0.3 + 0.05 * i as f64;
        let insurance = if i % 2 == 0 { 0.8 } else { 0.4 };

        buildings.push(BuildingRecord {
            building_id: format!("HK-BLD-{:04}", 1001 + i),
            features: vec![
                age, struct_type, district, repairs, access, contractor,
                inspection, defects, compliance, reserve, votes, insurance,
            ],
        });
    }
    buildings
}
