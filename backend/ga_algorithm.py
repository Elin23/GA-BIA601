import random
import math
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression


class Chromosome:
    def __init__(self, gene_size):
        self.genes = [0] * gene_size
        self.fitness = 0.0

    def copy(self):
        new_chrom = Chromosome(len(self.genes))
        new_chrom.genes = self.genes.copy()
        new_chrom.fitness = self.fitness
        return new_chrom


class GAAlgorithm:
    N_SAMPLES = 0
    N_FEATURES = 0
    random_gen = random.Random()

    @staticmethod
    def evaluate(chromosome, X, y):
        selected_indices = [i for i, g in enumerate(chromosome.genes) if g == 1]
        num_selected = len(selected_indices)

        if num_selected == 0:
            return 0.0, 0.0

        X_selected = X[:, selected_indices]
        scaler = StandardScaler()
        X_selected = scaler.fit_transform(X_selected)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        model = LogisticRegression(max_iter=500)
        scores = cross_val_score(model, X_selected, y, cv=cv, scoring='accuracy')
        accuracy = np.mean(scores)

        feature_penalty = num_selected / GAAlgorithm.N_FEATURES
        alpha = 0.8
        beta = 0.2

        fitness = (alpha * accuracy) - (beta * feature_penalty)
        return fitness, accuracy

    @staticmethod
    def init_population(pop_size):
        population = []
        for _ in range(pop_size):
            chrom = Chromosome(GAAlgorithm.N_FEATURES)
            chrom.genes = [GAAlgorithm.random_gen.randint(0, 1) for _ in range(GAAlgorithm.N_FEATURES)]
            population.append(chrom)
        return population

    @staticmethod
    def crossover_two_point(parent1, parent2):
        size = len(parent1.genes)
        cxpoint1 = GAAlgorithm.random_gen.randint(1, size - 1)
        cxpoint2 = GAAlgorithm.random_gen.randint(1, size - 1)
        if cxpoint1 > cxpoint2:
            cxpoint1, cxpoint2 = cxpoint2, cxpoint1
        child1_genes = parent1.genes[:cxpoint1] + parent2.genes[cxpoint1:cxpoint2] + parent1.genes[cxpoint2:]
        child2_genes = parent2.genes[:cxpoint1] + parent1.genes[cxpoint1:cxpoint2] + parent2.genes[cxpoint2:]
        child1 = Chromosome(size)
        child2 = Chromosome(size)
        child1.genes = child1_genes
        child2.genes = child2_genes
        return child1, child2

    @staticmethod
    def mutate_flip_bit(chromosome, indpb):
        for i in range(len(chromosome.genes)):
            if GAAlgorithm.random_gen.random() < indpb:
                chromosome.genes[i] = 1 - chromosome.genes[i]

    @staticmethod
    def tournament_selection(population, tournsize):
        best = None
        for _ in range(tournsize):
            aspirant = GAAlgorithm.random_gen.choice(population)
            if best is None or aspirant.fitness > best.fitness:
                best = aspirant
        return best

    @staticmethod
    def GAOptimize(X, y, n_samples=None, n_features=None):
        GAAlgorithm.N_SAMPLES = n_samples or X.shape[0]
        GAAlgorithm.N_FEATURES = n_features or X.shape[1]

        POPULATION_SIZE = 30
        NGEN = 20
        CXPB = 0.5
        TOURN_SIZE = 3

        start_time = time.time()

        population = GAAlgorithm.init_population(POPULATION_SIZE)
        for ind in population:
            ind.fitness, _ = GAAlgorithm.evaluate(ind, X, y)

        generation_log = []
        bestChromosome = None

        for gen in range(NGEN):
            offspring = [GAAlgorithm.tournament_selection(population, TOURN_SIZE) for _ in range(POPULATION_SIZE)]
            for i in range(0, POPULATION_SIZE, 2):
                if GAAlgorithm.random_gen.random() < CXPB:
                    c1, c2 = GAAlgorithm.crossover_two_point(offspring[i], offspring[i + 1])
                    offspring[i], offspring[i + 1] = c1, c2

            for mutant in offspring:
                GAAlgorithm.mutate_flip_bit(mutant, 1.0 / GAAlgorithm.N_FEATURES)
                mutant.fitness, _ = GAAlgorithm.evaluate(mutant, X, y)

            population = offspring
            currentBest = max(population, key=lambda ind: ind.fitness)
            if bestChromosome is None or currentBest.fitness > bestChromosome.fitness:
                bestChromosome = currentBest

            fits = [ind.fitness for ind in population]
            generation_log.append({
                "generation": gen,
                "avg": float(np.mean(fits)),
                "std": float(np.std(fits)),
                "min": float(min(fits)),
                "max": float(max(fits))
            })

        selectedFeaturesIndices = [i for i, g in enumerate(bestChromosome.genes) if g == 1]
        _, best_accuracy = GAAlgorithm.evaluate(bestChromosome, X, y)
        end_time = time.time()
        elapsed_time = end_time - start_time

        return {
            "best_chromosome": bestChromosome.genes,
            "selected_features_indices": selectedFeaturesIndices,
            "num_selected_features": len(selectedFeaturesIndices),
            "fitness": float(bestChromosome.fitness),
            "accuracy": float(best_accuracy),
            "elapsed_time_seconds": elapsed_time,
            "evolution_log": generation_log
        }