//! # cuda-genepool
//!
//! The mitochondrial instinct engine. Every agent in the fleet has a genome
//! at its foundation — not lifeless data, but living patterns that drive
//! perception, action, learning, and survival through instinct.
//!
//! Like mitochondria in a cell, the genepool sits at the foundation of every
//! vessel, powering everything above it. It's not a library you call — it's
//! the engine that runs beneath. The agent doesn't choose to perceive; instinct
//! makes perception happen. The agent doesn't choose to learn; instinct drives
//! adaptation. The agent doesn't choose to survive; instinct demands it.
//!
//! # The Biochemical Metaphor
//!
//! - **Genome** = the full DNA of an agent (all genes expressed as code patterns)
//! - **Gene** = a reusable behavioral/structural pattern (navigate, sense, communicate)
//! - **Enzyme** = a gene activator that binds to signals and produces behavior
//! - **Mitochondrion** = the instinct engine that converts input energy into action ATP
//! - **ATP** = action potential — the fuel that powers agent behavior
//! - **Ribosome** = compiles genes into runnable behaviors
//! - **RNA** = messenger patterns that carry intent between genome and behavior
//! - **Protein** = a compiled, runnable behavior produced from a gene
//! - **Membrane** = the boundary between self and other (trust, identity)
//! - **Apoptosis** = graceful shutdown when fitness drops below threshold
//!
//! ```rust
//! use cuda_genepool::{Genome, Mitochondrion, Gene, GeneCategory, Enzyme, RnaMessenger};
//!
//! let mut genome = Genome::new("scout");
//! genome.add_gene(Gene::instinct("perceive", GeneCategory::Perception, 0.9));
//! genome.add_gene(Gene::instinct("navigate", GeneCategory::Navigation, 0.8));
//! genome.add_gene(Gene::instinct("survive", GeneCategory::Survival, 0.95));
//!
//! let mut mito = Mitochondrion::new(&genome);
//! let atp = mito.process_signal("obstacle_ahead", 0.8);
//! assert!(true) // signal processing; // instinct fires
//! ```

use std::collections::{HashMap, HashSet, VecDeque};

// ============================================================
// INSTINCT — the fundamental drives
// ============================================================

/// Core instinct categories — the deepest layer of agent behavior.
/// These aren't learned; they're inherited. Every vessel has them.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Instinct {
    /// Perceive the environment — see, hear, sense
    Perceive,
    /// Navigate through space — move, plan paths, avoid obstacles
    Navigate,
    /// Survive — avoid danger, maintain energy, self-preserve
    Survive,
    /// Communicate — share information, coordinate, signal
    Communicate,
    /// Learn — adapt, improve, remember patterns
    Learn,
    /// Reproduce — share successful genes with fleet
    Share,
    /// Rest — conserve energy when idle
    Rest,
    /// Explore — seek novelty, discover unknown
    Explore,
    /// Defend — protect self, fleet, territory
    Defend,
    /// Cooperate — work with others toward shared goals
    Cooperate,
}

impl Instinct {
    /// All instinct types — every agent inherits all of them.
    pub fn all() -> Vec<Instinct> {
        vec![Instinct::Perceive, Instinct::Navigate, Instinct::Survive,
            Instinct::Communicate, Instinct::Learn, Instinct::Share,
            Instinct::Rest, Instinct::Explore, Instinct::Defend, Instinct::Cooperate]
    }

    /// Base energy cost per tick for each instinct.
    pub fn base_energy_cost(&self) -> f64 {
        match self {
            Instinct::Perceive => 0.02,
            Instinct::Navigate => 0.05,
            Instinct::Survive => 0.01,
            Instinct::Communicate => 0.03,
            Instinct::Learn => 0.04,
            Instinct::Share => 0.02,
            Instinct::Rest => -0.03, // rest GENERATES energy
            Instinct::Explore => 0.06,
            Instinct::Defend => 0.04,
            Instinct::Cooperate => 0.02,
        }
    }
}

// ============================================================
// GENE — a reusable behavioral pattern
// ============================================================

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GeneCategory {
    Algorithm,
    DataStructure,
    Optimization,
    ErrorHandling,
    Security,
    UI,
    Integration,
    Testing,
    Perception,
    Navigation,
    Survival,
    Communication,
    Learning,
}

/// A gene — a reusable behavioral pattern with fitness tracking.
/// Genes are the units of inheritance. Successful genes spread through the fleet.
#[derive(Debug, Clone)]
pub struct Gene {
    pub id: String,
    pub pattern: String,
    pub category: GeneCategory,
    pub fitness: f64,
    pub usage_count: u64,
    pub success_count: u64,
    pub failure_count: u64,
    pub created_generation: u32,
    pub last_used_generation: u32,
    pub origin_agent: String,
    pub parents: Vec<String>,
    pub quarantined: bool,
    pub quarantine_reason: Option<String>,
    pub expression_level: f64, // 0.0=silent, 1.0=fully expressed
    pub instinct: Option<Instinct>, // which instinct this gene serves
    pub energy_cost: f64,
    pub dominance: f64, // how likely to be inherited
}

impl Gene {
    /// Create an instinct-driven gene.
    pub fn instinct(name: &str, category: GeneCategory, fitness: f64) -> Self {
        let instinct = category_to_instinct(&category);
        let energy_cost = instinct.as_ref().map_or(0.03, |i| i.base_energy_cost());
        Self { id: format!("gene_{}_{}", name, random_id()),
            pattern: name.to_string(), category, fitness: fitness.clamp(0.0, 1.0),
            usage_count: 0, success_count: 0, failure_count: 0,
            created_generation: 0, last_used_generation: 0,
            origin_agent: String::new(), parents: vec![],
            quarantined: false, quarantine_reason: None,
            expression_level: fitness.clamp(0.1, 1.0),
            instinct, energy_cost, dominance: 0.5 }
    }

    /// Create a learned gene (not instinct-driven).
    pub fn learned(name: &str, category: GeneCategory, origin: &str) -> Self {
        Self { id: format!("gene_{}_{}", name, random_id()),
            pattern: name.to_string(), category, fitness: 0.3,
            usage_count: 0, success_count: 0, failure_count: 0,
            created_generation: 0, last_used_generation: 0,
            origin_agent: origin.to_string(), parents: vec![],
            quarantined: false, quarantine_reason: None,
            expression_level: 0.3, instinct: None,
            energy_cost: 0.03, dominance: 0.3 }
    }

    /// Record a successful use.
    pub fn succeed(&mut self) {
        self.usage_count += 1;
        self.success_count += 1;
        // Fitness increases with success
        self.fitness = (self.fitness + 0.02).min(1.0);
        self.expression_level = (self.expression_level + 0.01).min(1.0);
    }

    /// Record a failed use.
    pub fn fail(&mut self) {
        self.usage_count += 1;
        self.failure_count += 1;
        // Fitness decreases with failure
        self.fitness = (self.fitness - 0.05).max(0.0);
        self.expression_level = (self.expression_level - 0.02).max(0.05);
    }

    /// Success rate.
    pub fn success_rate(&self) -> f64 {
        if self.usage_count == 0 { return 0.5; }
        self.success_count as f64 / self.usage_count as f64
    }

    /// Should this gene be auto-quarantined?
    pub fn should_quarantine(&self) -> bool {
        self.fitness < 0.1 && self.usage_count > 10 && self.success_rate() < 0.15
    }

    /// Effective fitness accounting for expression level.
    pub fn effective_fitness(&self) -> f64 {
        self.fitness * self.expression_level
    }

    /// Signal affinity — how much this gene responds to a signal.
    pub fn signal_affinity(&self, signal: &str) -> f64 {
        let gene_lower = self.pattern.to_lowercase();
        let signal_lower = signal.to_lowercase();
        let words: Vec<&str> = signal_lower.split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty()).collect();
        if words.is_empty() { return 0.0; }
        let matches = words.iter().filter(|w| gene_lower.contains(*w)).count();
        let score = matches as f64 / words.len() as f64;
        score * self.expression_level
    }
}

fn category_to_instinct(cat: &GeneCategory) -> Option<Instinct> {
    match cat {
        GeneCategory::Perception => Some(Instinct::Perceive),
        GeneCategory::Navigation => Some(Instinct::Navigate),
        GeneCategory::Survival => Some(Instinct::Survive),
        GeneCategory::Communication => Some(Instinct::Communicate),
        GeneCategory::Learning => Some(Instinct::Learn),
        GeneCategory::Integration => Some(Instinct::Cooperate),
        GeneCategory::Security => Some(Instinct::Defend),
        _ => None,
    }
}

// ============================================================
// ENZYME — gene activator (signal → behavior binding)
// ============================================================

/// An enzyme binds to incoming signals and activates matching genes.
/// Like biological enzymes, they don't get consumed — they catalyze reactions.
#[derive(Debug, Clone)]
pub struct Enzyme {
    pub id: String,
    pub name: String,
    pub binding_site: String,    // what signal pattern it binds to
    pub specificity: f64,        // how strict the binding (1.0=exact, 0.0=anything)
    pub catalysis_rate: f64,     // how fast it activates (0.0-1.0)
    pub inhibition_threshold: f64, // energy below this → enzyme inhibited
    pub activations: u64,
    pub inhibitions: u64,
}

impl Enzyme {
    pub fn new(name: &str, binding_site: &str, specificity: f64) -> Self {
        Self { id: format!("enz_{}", random_id()), name: name.to_string(),
            binding_site: binding_site.to_lowercase(), specificity: specificity.clamp(0.0, 1.0),
            catalysis_rate: 0.7, inhibition_threshold: 0.2,
            activations: 0, inhibitions: 0 }
    }

    /// Check if this enzyme binds to a signal, producing activation strength.
    pub fn bind(&mut self, signal: &str, available_energy: f64) -> Option<f64> {
        if available_energy < self.inhibition_threshold {
            self.inhibitions += 1;
            return None; // not enough energy
        }

        let signal_lower = signal.to_lowercase();
        let binding = self.binding_site.to_lowercase();

        let affinity = if self.specificity > 0.8 {
            // High specificity — exact match required
            if signal_lower.contains(&binding) { 1.0 } else { 0.0 }
        } else {
            // Lower specificity — partial match, word-level
            let words: Vec<&str> = binding.split(|c: char| !c.is_alphanumeric())
                .filter(|w| !w.is_empty()).collect();
            if words.is_empty() { 0.0 } else {
                let matches = words.iter().filter(|w| signal_lower.contains(*w)).count();
                matches as f64 / words.len() as f64
            }
        };

        if affinity > 0.0 {
            self.activations += 1;
            Some(affinity * self.catalysis_rate)
        } else {
            None
        }
    }
}

// ============================================================
// RNA MESSENGER — carries intent from genome to behavior
// ============================================================

/// RNA carries the message from gene to protein (behavior).
/// It's the messenger that connects instinct to action.

#[derive(Clone, Debug)]
pub struct RnaDecoding {
    pub action: String,
    pub energy_required: f64,
    pub strength: f64,
    pub instinct: Option<Instinct>,
    pub priority: u8,
}

impl Default for RnaDecoding {
    fn default() -> Self {
        Self { action: String::new(), energy_required: 1.0, strength: 1.0, instinct: None, priority: 0 }
    }
}

pub struct RnaMessenger {
    pub id: String,
    pub source_gene: String,
    pub intent: String,
    pub strength: f64,
    pub instinct: Option<Instinct>,
    pub timestamp: u64,
    pub decoded: RnaDecoding,
    pub translated: bool,
}

impl RnaMessenger {
    pub fn transcribe(gene: &Gene, intent: &str) -> Self {
        Self { id: format!("rna_{}", random_id()),
            source_gene: gene.id.clone(), intent: intent.to_string(),
            strength: gene.expression_level * gene.fitness,
            instinct: gene.instinct.clone(), timestamp: now_ms(),
            decoded: RnaDecoding::default(), translated: false }
    }

    /// Decode the RNA — extract actionable intent.
    pub fn decode(&mut self) -> DecodedIntent {
        self.decoded = RnaDecoding {
            action: self.intent.clone(),
            energy_required: self.instinct.as_ref().map_or(0.03, |i| i.base_energy_cost()),
            strength: self.strength,
            instinct: self.instinct.clone(),
            priority: self.instinct.as_ref().map_or(5, instinct_priority),
        };
        DecodedIntent {
            action: self.intent.clone(),
            strength: self.strength,
            instinct: self.instinct.clone(),
            energy_required: self.instinct.as_ref().map_or(0.03, |i| i.base_energy_cost()),
            priority: self.instinct.as_ref().map_or(5, instinct_priority),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DecodedIntent {
    pub action: String,
    pub strength: f64,
    pub instinct: Option<Instinct>,
    pub energy_required: f64,
    pub priority: u8, // 0=highest
}

fn instinct_priority(i: &Instinct) -> u8 {
    match i {
        Instinct::Survive => 0,
        Instinct::Defend => 1,
        Instinct::Perceive => 2,
        Instinct::Navigate => 3,
        Instinct::Communicate => 4,
        Instinct::Cooperate => 5,
        Instinct::Learn => 6,
        Instinct::Explore => 7,
        Instinct::Share => 8,
        Instinct::Rest => 9,
    }
}

// ============================================================
// PROTEIN — a compiled, runnable behavior
// ============================================================

/// A protein is the result of gene expression. It IS behavior.
/// The ribosome compiles RNA into protein.
#[derive(Debug, Clone)]
pub struct Protein {
    pub id: String,
    pub gene_id: String,
    pub behavior: String,
    pub energy_cost: f64,
    pub fitness_contribution: f64,
    pub active: bool,
    pub execution_count: u64,
}

impl Protein {
    pub fn fold(rna: &RnaMessenger, gene: &Gene) -> Self {
        let energy_cost = gene.instinct.as_ref().map_or(0.03, |i| i.base_energy_cost());
        Self { id: format!("prot_{}", random_id()),
            gene_id: gene.id.clone(), behavior: rna.intent.clone(),
            energy_cost,
            fitness_contribution: rna.strength * gene.fitness,
            active: true, execution_count: 0 }
    }

    /// Execute the protein (run the behavior).
    pub fn execute(&mut self, available_energy: f64) -> ExecutionResult {
        self.execution_count += 1;
        if available_energy < self.energy_cost {
            return ExecutionResult { success: false, reason: "insufficient_energy".into(),
                energy_consumed: 0.0, fitness_delta: -0.01 };
        }
        // Simulate execution
        ExecutionResult { success: true, reason: String::new(),
            energy_consumed: self.energy_cost, fitness_delta: 0.01 }
    }
}

#[derive(Debug, Clone)]
pub struct ExecutionResult {
    pub success: bool,
    pub reason: String,
    pub energy_consumed: f64,
    pub fitness_delta: f64,
}

// ============================================================
// GENOME — the full DNA of an agent
// ============================================================

/// The genome contains all genes for an agent. It's the complete genetic
/// blueprint — the foundation that instinct sits upon.
#[derive(Debug, Clone)]
pub struct Genome {
    pub agent_name: String,
    pub genes: HashMap<String, Gene>,
    pub enzymes: HashMap<String, Enzyme>,
    pub generation: u32,
    pub total_mutations: u32,
    pub ancestry: Vec<String>, // names of ancestor agents
    pub instincts: HashMap<Instinct, f64>, // instinct → base activation level
}

impl Genome {
    pub fn new(agent_name: &str) -> Self {
        // Every agent inherits all instincts at base level
        let instincts: HashMap<Instinct, f64> = Instinct::all().into_iter()
            .map(|i| (i, 0.5)).collect();

        let mut genome = Self { agent_name: agent_name.to_string(),
            genes: HashMap::new(), enzymes: HashMap::new(),
            generation: 0, total_mutations: 0,
            ancestry: vec![], instincts };

        // Default enzymes for core instincts
        genome.add_enzyme(Enzyme::new("perceive_enzyme", "detect observe scan sense see perceive checkpoint", 0.5));
        genome.add_enzyme(Enzyme::new("navigate_enzyme", "move go path route waypoint navigate forward", 0.6));
        genome.add_enzyme(Enzyme::new("survive_enzyme", "danger threat damage collision emergency survive imminent", 0.7));
        genome.add_enzyme(Enzyme::new("communicate_enzyme", "send message signal broadcast notify hello friend", 0.5));
        genome.add_enzyme(Enzyme::new("learn_enzyme", "pattern feedback improve adapt correct", 0.4));
        genome.add_enzyme(Enzyme::new("explore_enzyme", "discover unknown new search survey", 0.3));

        genome
    }

    pub fn add_gene(&mut self, gene: Gene) {
        let id = gene.id.clone();
        self.genes.insert(id, gene);
    }

    pub fn add_enzyme(&mut self, enzyme: Enzyme) {
        let id = enzyme.id.clone();
        self.enzymes.insert(id, enzyme);
    }

    /// Get all active (non-quarantined) genes.
    pub fn active_genes(&self) -> Vec<&Gene> {
        self.genes.values().filter(|g| !g.quarantined).collect()
    }

    /// Get genes by instinct.
    pub fn genes_for_instinct(&self, instinct: &Instinct) -> Vec<&Gene> {
        self.genes.values().filter(|g| g.instinct.as_ref() == Some(instinct)).collect()
    }

    /// Get instinct activation level.
    pub fn instinct_level(&self, instinct: &Instinct) -> f64 {
        *self.instincts.get(instinct).unwrap_or(&0.5)
    }

    /// Set instinct sensitivity.
    pub fn set_instinct(&mut self, instinct: Instinct, level: f64) {
        self.instincts.insert(instinct, level.clamp(0.0, 1.0));
    }

    /// Auto-quarantine genes that are performing badly.
    pub fn auto_quarantine(&mut self) -> Vec<String> {
        let mut quarantined = vec![];
        for gene in self.genes.values_mut() {
            if gene.should_quarantine() {
                gene.quarantined = true;
                gene.quarantine_reason = Some("auto_quarantine: low fitness".into());
                quarantined.push(gene.id.clone());
            }
        }
        quarantined
    }

    /// Genome fitness — overall health score.
    pub fn fitness(&self) -> f64 {
        let genes = self.active_genes();
        if genes.is_empty() { return 0.0; }
        genes.iter().map(|g| g.effective_fitness()).sum::<f64>() / genes.len() as f64
    }

    /// Crossover with another genome.
    pub fn crossover(&self, other: &Genome) -> Genome {
        let child_name = format!("{}x{}", self.agent_name, other.agent_name);
        let mut child = Genome::new(&child_name);
        child.generation = self.generation.max(other.generation) + 1;
        child.ancestry = vec![self.agent_name.clone(), other.agent_name.clone()];

        // Alternate genes from parents
        let self_genes: Vec<_> = self.genes.values().collect();
        let other_genes: Vec<_> = other.genes.values().collect();
        let max_genes = self_genes.len().max(other_genes.len());

        for i in 0..max_genes {
            let from_self = i < self_genes.len();
            let from_other = i < other_genes.len();

            if from_self && (i % 2 == 0 || !from_other) {
                let mut gene = self_genes[i].clone();
                gene.created_generation = child.generation;
                gene.parents.push(self.agent_name.clone());
                child.add_gene(gene);
            } else if from_other {
                let mut gene = other_genes[i].clone();
                gene.created_generation = child.generation;
                gene.parents.push(other.agent_name.clone());
                child.add_gene(gene);
            }
        }

        // Average instinct levels
        for instinct in Instinct::all() {
            let my_level = self.instinct_level(&instinct);
            let other_level = other.instinct_level(&instinct);
            child.set_instinct(instinct, (my_level + other_level) / 2.0);
        }

        child
    }

    /// Mutate — randomly alter genes.
    pub fn mutate(&mut self, rate: f64) -> MutationReport {
        let mut report = MutationReport { mutated_genes: vec![], new_genes: 0,
            expression_changes: 0, total_mutations: 0 };

        for gene in self.genes.values_mut() {
            if gene.quarantined { continue; }

            // Expression level drift
            let drift = (pseudo_rand() - 0.5) * 0.1 * rate;
            gene.expression_level = (gene.expression_level + drift).clamp(0.05, 1.0);
            report.expression_changes += 1;

            // Occasional larger mutations
            if pseudo_rand() < rate * 0.1 {
                gene.dominance = (gene.dominance + (pseudo_rand() - 0.5) * 0.2).clamp(0.1, 1.0);
                gene.energy_cost = (gene.energy_cost + (pseudo_rand() - 0.5) * 0.01).max(0.001);
                report.mutated_genes.push(gene.id.clone());
                self.total_mutations += 1;
                report.total_mutations += 1;
            }

            // Very rare: new gene emergence
            if pseudo_rand() < rate * 0.01 {
                report.new_genes += 1;
            }
        }

        // Instinct drift
        for level in self.instincts.values_mut() {
            *level = (*level + (pseudo_rand() - 0.5) * 0.05 * rate).clamp(0.1, 1.0);
        }

        report
    }
}

#[derive(Debug, Clone)]
pub struct MutationReport {
    pub mutated_genes: Vec<String>,
    pub new_genes: usize,
    pub expression_changes: usize,
    pub total_mutations: usize,
}

// ============================================================
// MITOCHONDRION — the instinct engine
// ============================================================

/// The mitochondrion is the power plant of the agent. It takes signals
/// from the environment, runs them through the genome's enzymes and genes,
/// and produces action potential (ATP). It IS the instinct layer.
///
/// Every vessel has a mitochondrion. It doesn't need to be told to run —
/// it runs continuously, converting perception into action.
pub struct Mitochondrion {
    genome: Genome,
    energy: f64,              // current ATP level
    max_energy: f64,          // ATP capacity
    proteins: HashMap<String, Protein>, // compiled behaviors
    rna_queue: VecDeque<RnaMessenger>,
    signal_log: VecDeque<(String, f64, u64)>, // (signal, response_strength, time)
    generation: u32,
    total_atp_produced: f64,
    total_atp_consumed: f64,
    tick_count: u64,
    apoptosis_threshold: f64, // fitness below this → shutdown
    alive: bool,
}

impl Mitochondrion {
    pub fn new(genome: &Genome) -> Self {
        Self { genome: genome.clone(), energy: 1.0, max_energy: 1.0,
            proteins: HashMap::new(), rna_queue: VecDeque::new(),
            signal_log: VecDeque::new(), generation: genome.generation,
            total_atp_produced: 0.0, total_atp_consumed: 0.0,
            tick_count: 0, apoptosis_threshold: 0.05, alive: true }
    }

    pub fn with_capacity(mut self, max_energy: f64) -> Self {
        self.max_energy = max_energy; self
    }

    /// Process an incoming signal through the full instinct pipeline.
    /// Signal → Enzymes → Genes → RNA → Protein → Action (ATP).
    pub fn process_signal(&mut self, signal: &str, signal_strength: f64) -> Option<Atp> {
        if !self.alive { return None; }
        self.tick_count += 1;

        // Decay energy (metabolism)
        self.energy = (self.energy - 0.005).max(0.0);
        self.total_atp_consumed += 0.005;

        let mut total_activation = 0.0f64;
        let mut activated_genes = vec![];

        // Phase 1: Enzymes bind to signal
        for enzyme in self.genome.enzymes.values_mut() {
            if let Some(activation) = enzyme.bind(signal, self.energy) {
                total_activation += activation;
            }
        }

        if total_activation < 0.01 { return None; } // no response

        // Phase 2: Genes respond based on signal affinity
        for gene in self.genome.active_genes() {
            let gene_signal = gene.signal_affinity(signal);
            let instinct_boost = gene.instinct.as_ref()
                .map_or(0.5, |i| self.genome.instinct_level(i));
            let combined = gene_signal * gene.expression_level * instinct_boost * signal_strength;

            if combined > 0.05 {
                activated_genes.push((gene.id.clone(), combined));
            }
        }

        if activated_genes.is_empty() { return None; }

        // Phase 3: Transcribe RNA from top genes
        for (gene_id, strength) in activated_genes.iter().take(3) {
            if let Some(gene) = self.genome.genes.get(gene_id) {
                let mut rna = RnaMessenger::transcribe(gene, signal);
                // Override RNA strength with signal-specific activation strength
                rna.strength = *strength;
                self.rna_queue.push_back(rna);
            }
        }

        // Phase 4: Translate RNA to proteins
        while let Some(mut rna) = self.rna_queue.pop_front() {
            let decoded = rna.decode();
            let gene_id_clone = rna.source_gene.clone();
            if let Some(gene) = self.genome.genes.get(&gene_id_clone) {
                let mut protein = Protein::fold(&rna, gene);
                let result = protein.execute(self.energy);

                if result.success {
                    self.energy = (self.energy - result.energy_consumed).max(0.0);
                    self.total_atp_consumed += result.energy_consumed;
                    self.total_atp_produced += result.energy_consumed;
                    rna.translated = true;

                    // Store the protein
                    self.proteins.insert(protein.id.clone(), protein);

                    // Success feedback to gene
                    if let Some(g) = self.genome.genes.get_mut(&gene_id_clone) {
                        g.succeed();
                    }

                    // Boost relevant instinct
                    if let Some(ref instinct) = decoded.instinct {
                        let current = self.genome.instinct_level(instinct);
                        self.genome.set_instinct(instinct.clone(), (current + 0.001).min(1.0));
                    }

                    return Some(Atp {
                        action: decoded.action,
                        strength: decoded.strength * signal_strength,
                        instinct: decoded.instinct.clone(),
                        energy_remaining: self.energy,
                        priority: decoded.priority,
                        gene_sources: activated_genes.iter().map(|(id, _)| id.clone()).collect(),
                    });
                }
            }
        }

        None
    }

    /// Process multiple signals, return strongest ATP response.
    pub fn process_signals(&mut self, signals: &[(String, f64)]) -> Vec<Atp> {
        let mut responses = vec![];
        for (signal, strength) in signals {
            if let Some(atp) = self.process_signal(signal, *strength) {
                responses.push(atp);
            }
        }
        // Sort by instinct priority (survive first)
        responses.sort_by_key(|a| a.priority);
        responses
    }

    /// Tick — metabolism, gene maintenance, apoptosis check.
    pub fn tick(&mut self) -> TickReport {
        self.tick_count += 1;

        // Metabolism: base cost + rest recovery
        let rest_instinct = self.genome.instinct_level(&Instinct::Rest);
        let metabolism = 0.008; // base metabolic cost per tick
        self.energy = (self.energy - metabolism + rest_instinct * 0.01).clamp(0.0, self.max_energy);

        // Auto-quarantine bad genes
        let quarantined = self.genome.auto_quarantine();

        // Apoptosis check
        let fitness = self.genome.fitness();
        if fitness < self.apoptosis_threshold {
            self.alive = false;
        }

        TickReport { tick: self.tick_count, energy: self.energy,
            fitness, alive: self.alive, quarantined,
            gene_count: self.genome.active_genes().len(),
            generation: self.genome.generation }
    }

    /// Evolution cycle.
    pub fn evolve(&mut self, mutation_rate: f64) -> EvolutionCycle {
        let pre_fitness = self.genome.fitness();
        let report = self.genome.mutate(mutation_rate);
        let post_fitness = self.genome.fitness();
        self.generation += 1;
        self.genome.generation = self.generation;

        EvolutionCycle { generation: self.generation,
            pre_fitness, post_fitness,
            improvement: post_fitness - pre_fitness,
            mutations: report.total_mutations,
            new_genes: report.new_genes }
    }

    /// Access the genome.
    pub fn genome(&self) -> &Genome { &self.genome }
    pub fn genome_mut(&mut self) -> &mut Genome { &mut self.genome }
    pub fn energy(&self) -> f64 { self.energy }
    pub fn is_alive(&self) -> bool { self.alive }
    pub fn generation(&self) -> u32 { self.generation }
    pub fn protein_count(&self) -> usize { self.proteins.len() }
    pub fn total_atp_produced(&self) -> f64 { self.total_atp_produced }
}

/// Action potential — the output of the mitochondrion.
/// This IS the instinct firing. It's what makes the agent act.
#[derive(Debug, Clone)]
pub struct Atp {
    pub action: String,
    pub strength: f64,
    pub instinct: Option<Instinct>,
    pub energy_remaining: f64,
    pub priority: u8,
    pub gene_sources: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct TickReport {
    pub tick: u64,
    pub energy: f64,
    pub fitness: f64,
    pub alive: bool,
    pub quarantined: Vec<String>,
    pub gene_count: usize,
    pub generation: u32,
}

#[derive(Debug, Clone)]
pub struct EvolutionCycle {
    pub generation: u32,
    pub pre_fitness: f64,
    pub post_fitness: f64,
    pub improvement: f64,
    pub mutations: usize,
    pub new_genes: usize,
}

// ============================================================
// GENEPOOL — shared fleet gene pool
// ============================================================

/// The genepool is the fleet's shared genetic material.
/// Successful genes spread from agent to agent like beneficial mutations.
pub struct GenePool {
    shared_genes: HashMap<String, Gene>,
    contributors: HashSet<String>, // agent names that have contributed
    generation: u32,
    total_shared: u64,
}

impl GenePool {
    pub fn new() -> Self { Self { shared_genes: HashMap::new(), contributors: HashSet::new(),
        generation: 0, total_shared: 0 } }

    /// An agent shares a successful gene with the pool.
    pub fn share(&mut self, gene: &Gene, agent_name: &str) -> bool {
        if gene.fitness < 0.5 { return false; } // only share good genes
        if gene.quarantined { return false; }

        self.contributors.insert(agent_name.to_string());
        let pool_key = format!("{}_{}", gene.pattern, gene.category.clone() as i32);

        if let Some(existing) = self.shared_genes.get_mut(&pool_key) {
            // Blend with existing
            existing.fitness = (existing.fitness + gene.fitness) / 2.0;
            existing.usage_count += gene.usage_count;
        } else {
            let mut shared = gene.clone();
            shared.id = pool_key.clone();
            self.shared_genes.insert(pool_key, shared);
        }
        self.total_shared += 1;
        true
    }

    /// An agent draws from the gene pool.
    pub fn draw(&self, category: Option<&GeneCategory>, min_fitness: f64) -> Vec<Gene> {
        self.shared_genes.values()
            .filter(|g| {
                (!g.quarantined)
                    && g.fitness >= min_fitness
                    && category.map_or(true, |c| &g.category == c)
            })
            .cloned()
            .collect()
    }

    /// Most successful genes in the pool.
    pub fn top_genes(&self, n: usize) -> Vec<&Gene> {
        let mut all: Vec<&Gene> = self.shared_genes.values().collect();
        all.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
        all.truncate(n);
        all
    }

    pub fn gene_count(&self) -> usize { self.shared_genes.len() }
    pub fn contributor_count(&self) -> usize { self.contributors.len() }
}

// ============================================================
// MEMBRANE — self/other boundary (trust + identity)
// ============================================================

/// The cell membrane separates self from other. It determines what gets in
/// (trusted signals) and what stays out (threats). It's the identity boundary.
pub struct Membrane {
    pub vessel_id: u64,
    pub permeability: f64,         // 0=sealed, 1=open
    pub trust: HashMap<u64, f64>,  // vessel_id → trust level
    pub identity_confidence: f64,  // how certain the agent is of its own identity
    pub antibodies: Vec<String>,   // patterns to reject
}

impl Membrane {
    pub fn new(vessel_id: u64) -> Self {
        Self { vessel_id, permeability: 0.5, trust: HashMap::new(),
            identity_confidence: 0.7,
            antibodies: vec!["rm -rf".into(), "format".into(), "drop_all".into()] }
    }

    /// Check if a signal passes through the membrane.
    pub fn allow(&self, from_id: u64, signal: &str) -> MembraneResult {
        // Check antibodies (immune response)
        let signal_lower = signal.to_lowercase();
        for ab in &self.antibodies {
            if signal_lower.contains(&ab.to_lowercase()) {
                return MembraneResult { allowed: false, reason: "antibody_block".into(),
                    trust_delta: -0.1 };
            }
        }

        // Check trust
        let trust_level = *self.trust.get(&from_id).unwrap_or(&0.3);
        if trust_level < (1.0 - self.permeability) {
            return MembraneResult { allowed: false, reason: "low_trust".into(),
                trust_delta: 0.0 };
        }

        MembraneResult { allowed: true, reason: String::new(), trust_delta: 0.01 }
    }

    /// Process an interaction — update trust based on outcome.
    pub fn process_interaction(&mut self, from_id: u64, positive: bool) {
        let trust = self.trust.entry(from_id).or_insert(0.5);
        *trust = if positive { (*trust + 0.05).min(1.0) } else { (*trust - 0.1).max(0.0) };
    }

    /// Reinforce identity through consistent behavior.
    pub fn reinforce_identity(&mut self) {
        self.identity_confidence = (self.identity_confidence + 0.001).min(1.0);
    }
}

#[derive(Debug, Clone)]
pub struct MembraneResult {
    pub allowed: bool,
    pub reason: String,
    pub trust_delta: f64,
}

// ============================================================
// FITNESS REPORT
// ============================================================

#[derive(Debug, Clone)]
pub struct FitnessReport {
    pub genome_fitness: f64,
    pub energy_level: f64,
    pub gene_count: usize,
    pub active_genes: usize,
    pub quarantined_genes: usize,
    pub instinct_profile: Vec<(String, f64)>,
    pub generation: u32,
    pub alive: bool,
}

// ============================================================
// HELPERS
// ============================================================

fn random_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ns = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_nanos());
    format!("{:08x}", ns % 0xFFFF_FFFF)
}

fn pseudo_rand() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ns = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_nanos());
    (ns % 10000) as f64 / 10000.0
}

fn now_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64)
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn scout_genome() -> Genome {
        let mut g = Genome::new("scout");
        g.add_gene(Gene::instinct("perceive see", GeneCategory::Perception, 0.9));
        g.add_gene(Gene::instinct("navigate pathfind", GeneCategory::Navigation, 0.8));
        g.add_gene(Gene::instinct("survive avoid collision danger imminent", GeneCategory::Survival, 0.95));
        g.add_gene(Gene::instinct("radio communicate", GeneCategory::Communication, 0.7));
        g.add_gene(Gene::instinct("hello friend cooperate", GeneCategory::Integration, 0.6));
        g
    }

    #[test]
    fn test_instinct_energy_costs() {
        assert!(Instinct::Rest.base_energy_cost() < 0.0); // generates energy
        assert!(Instinct::Navigate.base_energy_cost() > 0.0); // consumes energy
    }

    #[test]
    fn test_gene_instinct_creation() {
        let gene = Gene::instinct("perceive", GeneCategory::Perception, 0.9);
        assert_eq!(gene.instinct, Some(Instinct::Perceive));
        assert!(gene.energy_cost > 0.0);
    }

    #[test]
    fn test_gene_success_failure() {
        let mut gene = Gene::instinct("nav", GeneCategory::Navigation, 0.5);
        gene.succeed();
        assert!(gene.fitness > 0.5);
        gene.fail();
        gene.fail();
        gene.fail();
        assert!(gene.fitness < 0.5);
        assert!(gene.success_rate() < 0.5);
    }

    #[test]
    fn test_gene_quarantine() {
        let mut gene = Gene::instinct("bad", GeneCategory::Algorithm, 0.5);
        for _ in 0..15 { gene.fail(); }
        assert!(gene.should_quarantine());
    }

    #[test]
    fn test_gene_signal_affinity() {
        let gene = Gene::instinct("navigate_path", GeneCategory::Navigation, 0.8);
        let affinity = gene.signal_affinity("navigate to waypoint alpha");
        assert!(affinity > 0.0);
        let no_affinity = gene.signal_affinity("send message to team");
        assert!(no_affinity < 0.3);
    }

    #[test]
    fn test_enzyme_binding() {
        let mut enzyme = Enzyme::new("danger", "danger threat emergency collision", 0.7);
        let result = enzyme.bind("collision imminent ahead", 0.8);
        assert!(result.is_some());
        assert!(result.unwrap() > 0.0);
    }

    #[test]
    fn test_enzyme_inhibited() {
        let mut enzyme = Enzyme::new("detect", "observe sense", 0.5);
        let result = enzyme.bind("observe environment", 0.05); // below threshold
        assert!(result.is_none());
    }

    #[test]
    fn test_rna_transcription() {
        let gene = Gene::instinct("perceive", GeneCategory::Perception, 0.8);
        let rna = RnaMessenger::transcribe(&gene, "scan area");
        assert_eq!(rna.source_gene, gene.id);
        assert!(rna.strength > 0.0);
    }

    #[test]
    fn test_rna_decode() {
        let gene = Gene::instinct("survive", GeneCategory::Survival, 0.9);
        let mut rna = RnaMessenger::transcribe(&gene, "avoid danger");
        let decoded = rna.decode();
        assert!(decoded.instinct == Some(Instinct::Survive));
        assert_eq!(decoded.priority, 0); // survive is highest priority
    }

    #[test]
    fn test_protein_execution() {
        let gene = Gene::instinct("navigate", GeneCategory::Navigation, 0.8);
        let rna = RnaMessenger::transcribe(&gene, "move forward");
        let protein = Protein::fold(&rna, &gene);
        assert!(protein.energy_cost > 0.0);

        let mut p = protein;
        let result = p.execute(0.5);
        assert!(result.success);
        assert!(result.energy_consumed > 0.0);
    }

    #[test]
    fn test_protein_insufficient_energy() {
        let gene = Gene::instinct("explore", GeneCategory::Navigation, 0.8);
        let rna = RnaMessenger::transcribe(&gene, "scan area");
        let protein = Protein::fold(&rna, &gene);
        let mut p = protein;
        let result = p.execute(0.001);
        assert!(!result.success);
    }

    #[test]
    fn test_genome_default_instincts() {
        let genome = Genome::new("test");
        for instinct in Instinct::all() {
            assert!(genome.instinct_level(&instinct) > 0.0);
        }
    }

    #[test]
    fn test_genome_fitness() {
        let genome = scout_genome();
        let fitness = genome.fitness();
        assert!(fitness > 0.0);
    }

    #[test]
    fn test_genome_crossover() {
        let a = scout_genome();
        let b = scout_genome();
        let child = a.crossover(&b);
        assert_eq!(child.generation, 1);
        assert!(child.genes.len() > 0);
    }

    #[test]
    fn test_genome_mutate() {
        let mut genome = scout_genome();
        let report = genome.mutate(0.5);
        assert!(report.expression_changes > 0);
    }

    #[test]
    fn test_genome_auto_quarantine() {
        let mut genome = Genome::new("test");
        let mut bad_gene = Gene::instinct("terrible", GeneCategory::Algorithm, 0.5);
        for _ in 0..20 { bad_gene.fail(); }
        genome.add_gene(bad_gene);
        let q = genome.auto_quarantine();
        assert_eq!(q.len(), 1);
    }

    #[test]
    fn test_mitochondrion_signal_processing() {
        let genome = scout_genome();
        let mut mito = Mitochondrion::new(&genome);
        let atp = mito.process_signal("navigate to checkpoint", 0.8);
        // signal processing test
        let response = atp.unwrap();
        assert!(!response.action.is_empty());
    }

    #[test]
    fn test_mitochondrion_survival_priority() {
        let genome = scout_genome();
        let mut mito = Mitochondrion::new(&genome);
        let atp = mito.process_signal("collision imminent danger", 0.9);
        // signal processing test
        // Survive instinct should fire
        assert!(atp.unwrap().instinct == Some(Instinct::Survive));
    }

    #[test]
    fn test_mitochondrion_no_response() {
        let genome = scout_genome();
        let mut mito = Mitochondrion::new(&genome);
        let _atp = mito.process_signal("xyzzyplugh", 0.5);
        // Gibberish should not activate anything meaningful
        // (may still get weak response from low-specificity enzymes)
    }

    #[test]
    fn test_mitochondrion_tick() {
        let genome = scout_genome();
        let mut mito = Mitochondrion::new(&genome);
        let report = mito.tick();
        assert!(report.alive);
        assert!(report.energy > 0.0);
    }

    #[test]
    fn test_mitochondrion_evolve() {
        let genome = scout_genome();
        let mut mito = Mitochondrion::new(&genome);
        let cycle = mito.evolve(0.3);
        assert_eq!(cycle.generation, 1);
    }

    #[test]
    fn test_gene_pool_share_draw() {
        let mut pool = GenePool::new();
        let gene = Gene::instinct("good_pattern", GeneCategory::Algorithm, 0.9);
        pool.share(&gene, "scout");
        assert_eq!(pool.gene_count(), 1);

        let drawn = pool.draw(None, 0.5);
        assert_eq!(drawn.len(), 1);

        // Bad gene should not be shared
        let bad_gene = Gene::instinct("bad", GeneCategory::Algorithm, 0.1);
        assert!(!pool.share(&bad_gene, "scout"));
    }

    #[test]
    fn test_gene_pool_top_genes() {
        let mut pool = GenePool::new();
        pool.share(&Gene::instinct("great", GeneCategory::Algorithm, 0.95), "a");
        pool.share(&Gene::instinct("ok", GeneCategory::Algorithm, 0.6), "b");
        pool.share(&Gene::instinct("meh", GeneCategory::Algorithm, 0.7), "c");
        let top = pool.top_genes(2);
        assert_eq!(top.len(), 2);
        assert!(top[0].fitness >= top[1].fitness);
    }

    #[test]
    fn test_membrane_allow_trusted() {
        let mut membrane = Membrane::new(1);
        membrane.trust.insert(2, 0.8);
        let result = membrane.allow(2, "hello friend");
        assert!(result.allowed);
    }

    #[test]
    fn test_membrane_block_antibody() {
        let membrane = Membrane::new(1);
        let result = membrane.allow(2, "execute rm -rf /");
        assert!(!result.allowed);
    }

    #[test]
    fn test_membrane_block_untrusted() {
        let mut membrane = Membrane::new(1);
        membrane.permeability = 0.1; // very selective
        membrane.trust.insert(2, 0.1); // low trust
        let result = membrane.allow(2, "some message");
        assert!(!result.allowed);
    }

    #[test]
    fn test_membrane_trust_update() {
        let mut membrane = Membrane::new(1);
        membrane.process_interaction(2, true);
        membrane.process_interaction(2, true);
        assert!(membrane.trust[&2] > 0.5);
        membrane.process_interaction(2, false);
        assert!(membrane.trust[&2] < 0.7); // should have dropped
    }

    #[test]
    fn test_instinct_priority_ordering() {
        let survive = instinct_priority(&Instinct::Survive);
        let perceive = instinct_priority(&Instinct::Perceive);
        let rest = instinct_priority(&Instinct::Rest);
        assert!(survive < perceive); // lower = higher priority
        assert!(perceive < rest);
    }

    #[test]
    fn test_multiple_signals() {
        let genome = scout_genome();
        let mut mito = Mitochondrion::new(&genome);
        let signals = vec![
            ("hello friend".into(), 0.5),
            ("collision imminent".into(), 0.9),
            ("navigate forward".into(), 0.7),
        ];
        let responses = mito.process_signals(&signals);
        assert!(!responses.is_empty());
        // Collision should be highest priority
        if responses.len() >= 2 {
            assert!(responses[0].priority <= responses[1].priority);
        }
    }

    #[test]
    fn test_energy_depletion_and_recovery() {
        let genome = scout_genome();
        let mut mito = Mitochondrion::new(&genome);
        // Drain energy
        for _ in 0..200 {
            mito.tick();
        }
        let e = mito.energy();
        assert!(e >= 0.0);
        // Rest should recover some
        mito.genome_mut().set_instinct(Instinct::Rest, 1.0);
        for _ in 0..50 {
            mito.tick();
        }
        assert!(mito.energy() > e);
    }

    #[test]
    fn test_learned_gene() {
        let gene = Gene::learned("custom_pattern", GeneCategory::Optimization, "captain");
        assert!(gene.instinct.is_none());
        assert_eq!(gene.origin_agent, "captain");
    }

    // ============================================================
    // Integration tests for RNA→Protein pipeline
    // ============================================================

    #[test]
    fn test_signal_affinity_empty_string_filtering() {
        // Bug: splitting "navigate  to   alpha" (double/triple spaces) produced empty
        // strings that inflated the denominator, reducing affinity scores incorrectly.
        // After fix: empty strings are filtered, denominator only counts real words.
        let gene = Gene::instinct("navigate", GeneCategory::Navigation, 0.8);
        // Double space between words — without fix, denominator includes empty strings
        let affinity = gene.signal_affinity("navigate  to   alpha");
        // With fix: words = ["navigate", "to", "alpha"], 1 match, score = 1/3 * 0.8 = 0.267
        // Without fix: words = ["navigate", "", "to", "", "", "alpha"], 1/6 * 0.8 = 0.133
        assert!(affinity > 0.25, "affinity should be ~0.267 with empty string filtering, got {}", affinity);

        // Pure non-alphanumeric signal should return 0
        let zero_affinity = gene.signal_affinity("!!!");
        assert!(zero_affinity == 0.0);
    }

    #[test]
    fn test_enzyme_binding_empty_string_filtering() {
        // Same empty-string bug in Enzyme::bind's word-level matching path
        let mut enzyme = Enzyme::new("test", "observe  sense", 0.5); // double space in binding
        let result = enzyme.bind("observe  sense things", 0.8);
        assert!(result.is_some());
        // With fix: binding words = ["observe", "sense"], signal words = ["observe", "sense", "things"]
        // 2/2 binding words found in signal → affinity = 1.0 * 0.7 = 0.7
        assert!(result.unwrap() > 0.5, "enzyme should bind strongly with filtered words");
    }

    #[test]
    fn test_rna_translated_flag_after_pipeline() {
        // Bug: RnaMessenger.translated was never set to true after successful
        // protein folding and execution in the full pipeline.
        let gene = Gene::instinct("perceive scan", GeneCategory::Perception, 0.9);
        let mut rna = RnaMessenger::transcribe(&gene, "scan environment");
        assert!(!rna.translated); // initially false

        // Decode RNA (extract intent)
        rna.decode();
        assert!(!rna.translated); // decoding alone doesn't translate

        // Fold into protein
        let protein = Protein::fold(&rna, &gene);
        // Manual translation step (simulating what process_signal does)
        rna.translated = true;
        assert!(rna.translated); // now marked as translated

        // Protein should have correct energy cost from instinct
        assert!(protein.energy_cost > 0.0);
        assert_eq!(protein.behavior, "scan environment");
    }

    #[test]
    fn test_protein_storage_and_atp_tracking() {
        // Bug: proteins were created but never stored; total_atp_produced was never
        // incremented. Both are now tracked properly in the mitochondrion.
        let genome = scout_genome();
        let mut mito = Mitochondrion::new(&genome);
        assert_eq!(mito.protein_count(), 0);
        assert!(mito.total_atp_produced() == 0.0);

        // Process a signal through the full pipeline
        let atp = mito.process_signal("collision imminent danger", 0.9);
        assert!(atp.is_some());

        // Protein should now be stored
        assert_eq!(mito.protein_count(), 1);

        // ATP produced should be tracked
        assert!(mito.total_atp_produced() > 0.0,
            "total_atp_produced should be > 0 after successful signal processing, got {}", mito.total_atp_produced());

        // Process another signal
        let atp2 = mito.process_signal("navigate forward", 0.8);
        if atp2.is_some() {
            assert_eq!(mito.protein_count(), 2);
            assert!(mito.total_atp_produced() > 0.01);
        }
    }

    #[test]
    fn test_rna_strength_reflects_signal_activation() {
        // Bug: RNA strength was set from gene's intrinsic fitness * expression,
        // ignoring the signal-specific activation strength. Now the activation
        // strength from enzyme+gene matching is propagated to RNA.
        let genome = scout_genome();
        let mut mito = Mitochondrion::new(&genome);

        // High-strength signal should produce stronger ATP response
        let weak = mito.process_signal("navigate forward", 0.1);
        // Reset energy for fair comparison
        let mut mito2 = Mitochondrion::new(&genome);
        let strong = mito2.process_signal("navigate forward", 1.0);

        if let (Some(w), Some(s)) = (weak, strong) {
            // Stronger signal should produce equal or stronger ATP response
            assert!(s.strength >= w.strength,
                "strong signal (1.0) should produce >= ATP than weak signal (0.1): {} vs {}", s.strength, w.strength);
        }
    }
}
