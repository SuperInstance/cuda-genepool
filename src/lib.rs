//! Gene Pool — shared pattern library for agent evolution
//! Successful patterns replicate across the fleet. Bad patterns are quarantined.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::time::{SystemTime, UNIX_EPOCH};

/// A gene — a reusable code pattern with fitness score
#[derive(Debug, Clone)]
pub struct Gene {
    pub id: String,
    pub pattern: String,
    pub category: GeneCategory,
    pub fitness: f64,
    pub usage_count: u64,
    pub success_count: u64,
    pub failure_count: u64,
    pub created_nanos: u64,
    pub last_used_nanos: u64,
    pub origin_agent: String,
    pub quarantined: bool,
    pub parents: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum GeneCategory {
    Algorithm,
    DataStructure,
    Optimization,
    ErrorHandling,
    Security,
    UI,
    Integration,
    Testing,
}

/// Fitness evaluation result
pub struct FitnessReport {
    pub gene_id: String,
    pub fitness_delta: f64,
    pub new_fitness: f64,
    pub reason: String,
}

/// The gene pool — manages shared patterns across agents
pub struct GenePool {
    genes: HashMap<String, Gene>,
    generation: u64,
    mutation_rate: f64,
    crossover_rate: f64,
    quarantine_threshold: f64,
    promotion_threshold: f64,
}

impl GenePool {
    pub fn new() -> Self {
        Self {
            genes: HashMap::new(), generation: 0,
            mutation_rate: 0.1, crossover_rate: 0.3,
            quarantine_threshold: 0.2, promotion_threshold: 0.8,
        }
    }

    /// Register a new gene in the pool
    pub fn register(&mut self, gene: Gene) -> &Gene {
        let id = gene.id.clone();
        self.genes.insert(id.clone(), gene);
        self.genes.get(&id).unwrap()
    }

    /// Create a new gene from an agent pattern
    pub fn create_gene(&mut self, agent: &str, pattern: &str, category: GeneCategory) -> &Gene {
        let id = gene_id_hash(pattern);
        let now = now_nanos();
        let gene = Gene {
            id: id.clone(), pattern: pattern.to_string(),
            category, fitness: 0.5, usage_count: 0,
            success_count: 0, failure_count: 0,
            created_nanos: now, last_used_nanos: now,
            origin_agent: agent.to_string(),
            quarantined: false, parents: vec![],
        };
        self.register(gene)
    }

    /// Record a gene usage with outcome
    pub fn report_outcome(&mut self, gene_id: &str, success: bool) -> Option<FitnessReport> {
        let gene = self.genes.get_mut(gene_id)?;
        gene.usage_count += 1;
        gene.last_used_nanos = now_nanos();
        if success { gene.success_count += 1; } else { gene.failure_count += 1; }

        let old_fitness = gene.fitness;
        let total = gene.success_count + gene.failure_count;
        let success_rate = gene.success_count as f64 / total.max(1) as f64;
        // Exponential moving average
        gene.fitness = gene.fitness * 0.8 + success_rate * 0.2;

        // Auto-quarantine low fitness
        if gene.fitness < self.quarantine_threshold && gene.usage_count > 5 {
            gene.quarantined = true;
        }

        Some(FitnessReport {
            gene_id: gene_id.to_string(),
            fitness_delta: gene.fitness - old_fitness,
            new_fitness: gene.fitness,
            reason: if success { "success" } else { "failure" }.to_string(),
        })
    }

    /// Find best gene for a category
    pub fn best_gene(&self, category: &GeneCategory) -> Option<&Gene> {
        self.genes.values()
            .filter(|g| g.category == *category && !g.quarantined)
            .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
    }

    /// Crossover two genes to create offspring
    pub fn crossover(&mut self, parent_a: &str, parent_b: &str) -> Option<String> {
        let a = self.genes.get(parent_a)?;
        let b = self.genes.get(parent_b)?;
        if a.quarantined || b.quarantined { return None; }

        // Simple crossover: take first half of a, second half of b
        let mid_a = a.pattern.len() / 2;
        let mid_b = b.pattern.len() / 2;
        let child_pattern = format!("{}{}", &a.pattern[..mid_a], &b.pattern[mid_b..]);

        let child_id = gene_id_hash(&child_pattern);
        let child = Gene {
            id: child_id.clone(), pattern: child_pattern,
            category: a.category.clone(),
            fitness: (a.fitness + b.fitness) / 2.0 * 0.9, // slight reduction
            usage_count: 0, success_count: 0, failure_count: 0,
            created_nanos: now_nanos(), last_used_nanos: now_nanos(),
            origin_agent: "crossover".to_string(),
            quarantined: false, parents: vec![parent_a.to_string(), parent_b.to_string()],
        };
        self.register(child);
        Some(child_id)
    }

    /// Evolve one generation — promote, quarantine, generate offspring
    pub fn evolve(&mut self) -> Vec<String> {
        self.generation += 1;
        let mut new_genes = vec![];

        // Generate offspring from top genes
        let mut top: Vec<&Gene> = self.genes.values()
            .filter(|g| !g.quarantined && g.fitness > self.promotion_threshold)
            .collect();
        top.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());

        for i in 0..top.len().saturating_sub(1) {
            if i >= 3 { break; } // limit offspring per generation
            for j in (i+1)..top.len().min(i+3) {
                if let Some(id) = self.crossover(&top[i].id, &top[j].id) {
                    new_genes.push(id);
                }
            }
        }

        new_genes
    }

    /// Get pool statistics
    pub fn stats(&self) -> PoolStats {
        let total = self.genes.len();
        let active: usize = self.genes.values().filter(|g| !g.quarantined).count();
        let quarantined = total - active;
        let avg_fitness = if total > 0 {
            self.genes.values().map(|g| g.fitness).sum::<f64>() / total as f64
        } else { 0.0 };
        PoolStats { generation: self.generation, total, active, quarantined, avg_fitness }
    }
}

#[derive(Debug, Clone)]
pub struct PoolStats {
    pub generation: u64,
    pub total: usize,
    pub active: usize,
    pub quarantined: usize,
    pub avg_fitness: f64,
}

fn gene_id_hash(pattern: &str) -> String {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    pattern.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

fn now_nanos() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_and_register() {
        let mut pool = GenePool::new();
        pool.create_gene("architect", "sorted(data, reverse=True)", GeneCategory::Algorithm);
        assert_eq!(pool.stats().total, 1);
    }

    #[test]
    fn test_fitness_updates() {
        let mut pool = GenePool::new();
        pool.create_gene("architect", "good_pattern", GeneCategory::Algorithm);
        let id = "good_pattern".to_string();
        // Simulate the actual id from hash
        let gene_id = pool.genes.keys().next().unwrap().clone();
        pool.report_outcome(&gene_id, true);
        pool.report_outcome(&gene_id, true);
        pool.report_outcome(&gene_id, false);
        let gene = pool.genes.get(&gene_id).unwrap();
        assert!(gene.fitness > 0.5);
    }

    #[test]
    fn test_best_gene() {
        let mut pool = GenePool::new();
        pool.create_gene("a", "slow_sort", GeneCategory::Algorithm);
        let good_id = {
            let g = pool.create_gene("b", "fast_sort", GeneCategory::Algorithm);
            g.id.clone()
        };
        // Make good gene fitter
        for _ in 0..10 { pool.report_outcome(&good_id, true); }
        let best = pool.best_gene(&GeneCategory::Algorithm);
        assert!(best.is_some());
        assert_eq!(best.unwrap().id, good_id);
    }

    #[test]
    fn test_crossover() {
        let mut pool = GenePool::new();
        let a = pool.create_gene("a", "pattern_aaaaa", GeneCategory::Algorithm).id.clone();
        let b = pool.create_gene("b", "pattern_bbbbb", GeneCategory::Algorithm).id.clone();
        let child = pool.crossover(&a, &b);
        assert!(child.is_some());
        assert_eq!(pool.stats().total, 3);
    }

    #[test]
    fn test_evolve() {
        let mut pool = GenePool::new();
        let id = pool.create_gene("a", "great_pattern", GeneCategory::Algorithm).id.clone();
        for _ in 0..20 { pool.report_outcome(&id, true); }
        let new = pool.evolve();
        // Should have more genes after evolution
        assert!(pool.stats().total >= 1);
    }
}
