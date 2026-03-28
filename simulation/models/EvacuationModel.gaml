/**
 * EvacuationModel.gaml
 * Agent-based evacuation simulation for RespondOR-EvacuationRoute.
 *
 * Agents:
 *   - EvacueeAgent:   represents a group of people evacuating from a village
 *   - ShelterAgent:   represents an evacuation shelter with limited capacity
 *   - HazardAgent:    represents a static hazard field
 *
 * Movement model:
 *   - Evacuees move toward target shelter along pre-computed routes
 *   - Speed affected by congestion (BPR function) and hazard proximity
 *
 * Congestion model:
 *   - Road segment flow tracked globally
 *   - Travel speed reduced when flow > capacity
 *
 * Outputs (monitors):
 *   - total_saved: evacuees who reached shelter
 *   - total_delayed: evacuees still en-route at max steps
 *   - total_failed: evacuees who could not be assigned
 *   - evacuation_ratio: fraction of total pop evacuated
 *   - avg_evacuation_time: average time (steps) to shelter
 *   - worst_evacuation_time: maximum time (steps) to shelter
 */

model EvacuationSimulation

global {
    // === Input file parameters ===
    string villages_file      <- "inputs/villages.csv";
    string shelters_file      <- "inputs/shelters.csv";
    string routes_file        <- "inputs/routes.csv";
    string sim_config_file    <- "inputs/sim_config.json";
    int    run_id             <- 0;

    // === Simulation parameters ===
    float  time_step_min      <- 1.0;           // minutes per step
    float  hazard_radius_m    <- 5000.0;        // hazard effect radius
    float  base_speed_kmh     <- 30.0;          // base evacuation speed
    float  panic_factor       <- 1.2;           // speed reduction near hazard
    float  bpr_alpha          <- 0.15;          // BPR congestion alpha
    float  bpr_beta           <- 4.0;           // BPR congestion beta
    int    road_capacity_veh  <- 600;           // default road capacity (veh/hr)

    // === Statistics ===
    int    total_saved         <- 0;
    int    total_delayed       <- 0;
    int    total_failed        <- 0;
    float  evacuation_ratio    <- 0.0;
    float  avg_evacuation_time <- 0.0;
    float  worst_evacuation_time <- 0.0;

    // === Internal ===
    list<float> saved_times <- [];
    map<string, int> road_flow_map <- map<string, int>([]);

    init {
        write "=== Evacuation Simulation Run " + run_id + " ===";

        // Load shelters first (evacuees need shelter refs)
        create ShelterAgent from_csv shelters_file
            with: [
                shelter_id::  string(read("shelter_id")),
                shelter_name: string(read("name")),
                location:     { float(read("lon")), float(read("lat")) },
                max_capacity: int(read("capacity")),
                current_load: 0
            ];

        write "Loaded " + length(ShelterAgent) + " shelters";

        // Load villages and create evacuee groups
        matrix routes_data <- matrix(csv_file(routes_file, true));
        // Build route lookup: village_id -> {shelter_id, distance, time, pop}
        map<string, map<string, float>> route_lookup <- map<string, map<string, float>>([]);

        loop r over: rows_list(routes_data) {
            string vid      <- string(r[0]);
            string sid      <- string(r[1]);
            float  dist_km  <- float(r[2]);
            float  time_min <- float(r[3]);
            int    pop      <- int(r[4]);
            float  risk     <- float(r[5]);

            route_lookup[vid] <- map<string, float>([
                "shelter_id"::   float(0),    // store as index
                "distance_km"::  dist_km,
                "travel_time"::  time_min,
                "population"::   float(pop),
                "risk"::         risk
            ]);
            // Save shelter_id as string in a separate map
        }

        // Create evacuee agents from villages
        matrix villages_data <- matrix(csv_file(villages_file, true));
        loop v over: rows_list(villages_data) {
            string vid   <- string(v[0]);
            string vname <- string(v[1]);
            float  vlat  <- float(v[2]);
            float  vlon  <- float(v[3]);
            int    vpop  <- int(v[4]);

            // Find assigned shelter from routes
            if (route_lookup contains_key vid) {
                map route_info <- route_lookup[vid];
                float pop_assign <- route_info["population"];
                float dist       <- route_info["distance_km"];
                float travel     <- route_info["travel_time"];
                float risk       <- route_info["risk"];

                if (pop_assign > 0) {
                    create EvacueeAgent {
                        village_id      <- vid;
                        village_name    <- vname;
                        location        <- { vlon, vlat };
                        origin          <- location;
                        population_size <- int(pop_assign);
                        target_distance_km <- dist;
                        planned_travel_min <- travel;
                        route_risk      <- risk;
                        // Find target shelter
                        // (simplified: pick nearest available shelter)
                        target_shelter  <- one_of(ShelterAgent where (each.current_load < each.max_capacity));
                        status          <- "evacuating";
                    }
                }
            } else {
                total_failed <- total_failed + vpop;
            }
        }

        write "Created " + length(EvacueeAgent) + " evacuee groups";
        write "Failed to assign: " + total_failed;
    }

    reflex update_statistics when: every(10 #cycle) {
        total_saved    <- EvacueeAgent count (each.status = "saved");
        total_delayed  <- EvacueeAgent count (each.status = "evacuating");
        total_failed   <- total_failed;

        int total_pop <- sum(EvacueeAgent collect each.population_size) + total_failed;
        if (total_pop > 0) {
            evacuation_ratio <- float(total_saved) / float(total_pop);
        }

        if (!empty(saved_times)) {
            avg_evacuation_time   <- mean(saved_times) * time_step_min;
            worst_evacuation_time <- max(saved_times) * time_step_min;
        }
    }

    reflex end_check when: cycle = 500 {
        total_delayed <- EvacueeAgent count (each.status = "evacuating");
        write "=== Final Statistics ===";
        write "  Saved:    " + total_saved;
        write "  Delayed:  " + total_delayed;
        write "  Failed:   " + total_failed;
        write "  Ratio:    " + evacuation_ratio;
        write "  Avg time: " + avg_evacuation_time + " min";
        do pause;
    }
}

// =================================================================== //
// EVACUEE AGENT
// =================================================================== //
species EvacueeAgent {
    string  village_id;
    string  village_name;
    point   origin;
    int     population_size;
    float   target_distance_km;
    float   planned_travel_min;
    float   route_risk;
    string  status;              // "evacuating" | "saved" | "failed"
    int     steps_to_shelter;
    ShelterAgent target_shelter;
    int     steps_elapsed <- 0;

    float effective_speed_kmh {
        // Base speed reduced by congestion and hazard
        float congestion_factor <- 1.0;
        // BPR: t = t0 * (1 + alpha * (flow/cap)^beta)
        // Simplified: global congestion proxy
        float flow_ratio <- 0.5;  // placeholder; connect to road_flow_map
        congestion_factor <- 1.0 + bpr_alpha * (flow_ratio ^ bpr_beta);

        // Hazard proximity effect
        float hazard_penalty <- 1.0 + route_risk * 0.5;

        return base_speed_kmh / (congestion_factor * hazard_penalty);
    }

    reflex move when: status = "evacuating" {
        steps_elapsed <- steps_elapsed + 1;

        // Compute actual travel time accounting for speed
        float eff_speed <- effective_speed_kmh;
        float travel_time_steps <- (target_distance_km / eff_speed) * (60.0 / time_step_min);

        if (float(steps_elapsed) >= travel_time_steps) {
            do arrive;
        }
    }

    action arrive {
        if (target_shelter != nil and target_shelter.current_load < target_shelter.max_capacity) {
            target_shelter.current_load <- target_shelter.current_load + population_size;
            status <- "saved";
            add float(steps_elapsed) to: saved_times;
            ask world { total_saved <- total_saved + myself.population_size; }
        } else {
            // Try another shelter
            ShelterAgent alt <- one_of(ShelterAgent where (
                each.current_load + myself.population_size <= each.max_capacity
            ));
            if (alt != nil) {
                target_shelter <- alt;
                // Reset with extra travel time penalty
                steps_elapsed <- int(steps_elapsed * 0.7);
            } else {
                status <- "failed";
                ask world { total_failed <- total_failed + myself.population_size; }
            }
        }
    }

    aspect base {
        color c <- status = "saved" ? #green : (status = "failed" ? #red : #blue);
        draw circle(200) color: c;
    }
}

// =================================================================== //
// SHELTER AGENT
// =================================================================== //
species ShelterAgent {
    string  shelter_id;
    string  shelter_name;
    int     max_capacity;
    int     current_load <- 0;

    float utilization {
        return max_capacity > 0 ? float(current_load) / float(max_capacity) : 0.0;
    }

    aspect base {
        // Color by utilization: green=empty, red=full
        float util <- utilization();
        int r <- int(min(255, 2 * util * 255));
        int g <- int(min(255, 2 * (1.0 - util) * 255));
        color c <- rgb(r, g, 0);
        draw square(400) color: c border: #black;
    }
}

// =================================================================== //
// EXPERIMENTS
// =================================================================== //
experiment EvacuationExperiment type: gui {
    parameter "Villages file"   var: villages_file   category: "Input Files";
    parameter "Shelters file"   var: shelters_file   category: "Input Files";
    parameter "Routes file"     var: routes_file     category: "Input Files";
    parameter "Config file"     var: sim_config_file category: "Input Files";
    parameter "Run ID"          var: run_id          category: "Settings";
    parameter "Base speed (km/h)" var: base_speed_kmh category: "Model";
    parameter "BPR alpha"       var: bpr_alpha       category: "Model";
    parameter "BPR beta"        var: bpr_beta        category: "Model";
    parameter "Time step (min)" var: time_step_min   category: "Model";

    output {
        monitor "Saved"          value: total_saved;
        monitor "Delayed"        value: total_delayed;
        monitor "Failed"         value: total_failed;
        monitor "Evacuation %"   value: evacuation_ratio * 100.0;
        monitor "Avg time (min)" value: avg_evacuation_time;
        monitor "Worst time(min)" value: worst_evacuation_time;

        display EvacuationMap type: opengl {
            chart "Evacuation Progress" type: series {
                data "Saved" value: total_saved color: #green;
                data "Delayed" value: total_delayed color: #orange;
                data "Failed" value: total_failed color: #red;
            }
        }
    }
}

experiment BatchHeadless type: batch {
    parameter "Villages file"   var: villages_file;
    parameter "Shelters file"   var: shelters_file;
    parameter "Routes file"     var: routes_file;
    parameter "Config file"     var: sim_config_file;
    parameter "Run ID"          var: run_id among: [0,1,2,3,4];

    method exhaustive;

    permanent {
        monitor "total_saved"            value: total_saved;
        monitor "total_delayed"          value: total_delayed;
        monitor "total_failed"           value: total_failed;
        monitor "evacuation_ratio"       value: evacuation_ratio;
        monitor "avg_evacuation_time"    value: avg_evacuation_time;
        monitor "worst_evacuation_time"  value: worst_evacuation_time;
    }
}
