# cuda-genepool

**The mitochondrial instinct engine - the foundational power plant of every agent.**

> The agent doesn't choose to perceive. Instinct makes perception happen.

## What It Does

`cuda-genepool` is the biological foundation that makes every agent *alive*. It implements the full biological pipeline:

### The Pipeline

`Environment -> Sensors -> Membrane -> Enzymes -> Genes -> RNA -> Proteins -> ATP -> Execute`

- **Membrane** - Self/other boundary with antibody security
- **Enzymes** - Signal-to-gene binding (environmental triggers)
- **Genes** - Behavioral patterns with fitness tracking and auto-quarantine
- **RNA Messenger** - Gene-to-behavior translation
- **Protein** - Compiled, executable behavior
- **Mitochondrion** - ATP generation/consumption pipeline

### Instincts (10 built-in)

Each instinct has a priority, energy cost, and circadian modulation:
- **Survive** (priority 10) - HALT, TRAP opcodes
- **Perceive** (priority 8) - IO_READ, LOAD, CMP
- **Navigate** (priority 7) - JMP, CALL, RET
- **Communicate** (priority 6) - TELL, BROADCAST
- **Learn** (priority 5) - BOX, MOV
- **Defend** (priority 9) - REGION_GUARD, VERIFY
- **Rest** (priority 1) - Generates ATP (energy=-1.0)
- **Reproduce** (priority 3) - Gene crossover and sharing
- **Adapt** (priority 4) - Mutation and epigenetic modification
- **Cooperate** (priority 2) - GenePool sharing

### Key Mechanics

- **Gene auto-quarantine**: Fitness drops below 0.1 after 10+ uses with <15% success
- **Membrane antibodies**: "rm -rf", "format", "drop_all" blocked at the boundary
- **GenePool sharing**: Only genes above 0.5 fitness propagate
- **Apoptosis**: Triggered when fitness < 0.1 for patience_ticks consecutive cycles

## Ecosystem Integration

- `cuda-biology` - Higher-level BiologicalAgent using this engine
- `cuda-neurotransmitter` - Modulates instinct strength
- `cuda-energy` - ATP budgets and circadian rhythm
- `flux-runtime-c` - Instinct opcodes map to VM instructions
- `cuda-instruction-set` - 80 opcodes including instinct operations

## See Also

- [cuda-biology](https://github.com/Lucineer/cuda-biology) - Biological agent
- [cuda-energy](https://github.com/Lucineer/cuda-energy) - ATP and circadian
- [cuda-neurotransmitter](https://github.com/Lucineer/cuda-neurotransmitter) - Synapse modulation
- [flux-runtime-c](https://github.com/Lucineer/flux-runtime-c) - C VM executing instinct opcodes

## License

MIT OR Apache-2.0