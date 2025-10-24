import random
import math
import numpy as np


class Chromosome:
    def __init__(self, gene_size):
        self.genes = [0] * gene_size
        self.fitness = 0.0

    def __str__(self):
        return f"Genes: {self.genes}, Fitness: {self.fitness:.4f}"

    def copy(self):
        new_chrom = Chromosome(len(self.genes))
        new_chrom.genes = self.genes.copy()
        new_chrom.fitness = self.fitness
        return new_chrom



class GAAlgorithm:
    N_SAMPLES = 0
    N_FEATURES = 0
    featureScores = []
    random_gen = random.Random()

    @staticmethod
    def calculate_feature_scores(X, y):
        scores = [0.0] * GAAlgorithm.N_FEATURES
        for i in range(GAAlgorithm.N_FEATURES):
            feature_column = [X[j][i] for j in range(GAAlgorithm.N_SAMPLES)]
            meanX = np.mean(feature_column)
            meanY = np.mean(y)

            covariance = 0.0
            stdDevX = 0.0
            stdDevY = 0.0

            for j in range(GAAlgorithm.N_SAMPLES):
                covariance += (feature_column[j] - meanX) * (y[j] - meanY)
                stdDevX += (feature_column[j] - meanX) ** 2
                stdDevY += (y[j] - meanY) ** 2

            correlation = covariance / (math.sqrt(stdDevX) * math.sqrt(stdDevY))
            scores[i] = abs(correlation)
        return scores

    @staticmethod
    def evaluate(chromosome):
        selected_indices = [i for i, g in enumerate(chromosome.genes) if g == 1]
        num_selected = len(selected_indices)

        if num_selected == 0:
            return 0.0

        statistical_score = sum(GAAlgorithm.featureScores[i] for i in selected_indices)

        weightScore = 1.0
        weightFeatures = 0.5
        normalizedNumFeatures = num_selected / GAAlgorithm.N_FEATURES

        return (weightScore * statistical_score) - (weightFeatures * normalizedNumFeatures)

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
        cxpoint2 = GAAlgorithm.random_gen.randint(1, size - 2)

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
    def GAOptimize(X, y, n_samples=1000, n_features=20):
        GAAlgorithm.N_SAMPLES = n_samples
        GAAlgorithm.N_FEATURES = n_features
        result = ""
        result += "--- Starting Genetic Algorithm\n"

        GAAlgorithm.featureScores = GAAlgorithm.calculate_feature_scores(X, y)
        result += "Pre-computed feature scores: "
        result += ", ".join([f"{s:.4f}" for s in GAAlgorithm.featureScores]) + "\n"

        # GA parameters
        POPULATION_SIZE = 100
        NGEN = 50
        CXPB = 0.5
        MUTPB = 0.2
        TOURN_SIZE = 3

        population = GAAlgorithm.init_population(POPULATION_SIZE)
        bestChromosome = None

        for ind in population:
            ind.fitness = GAAlgorithm.evaluate(ind)

        for gen in range(NGEN):
            offspring = [GAAlgorithm.tournament_selection(population, TOURN_SIZE) for _ in range(POPULATION_SIZE)]

            for i in range(0, POPULATION_SIZE, 2):
                if GAAlgorithm.random_gen.random() < CXPB:
                    child1, child2 = GAAlgorithm.crossover_two_point(offspring[i], offspring[i + 1])
                    offspring[i] = child1
                    offspring[i + 1] = child2

            for mutant in offspring:
                GAAlgorithm.mutate_flip_bit(mutant, 1.0 / GAAlgorithm.N_FEATURES)

            for ind in offspring:
                ind.fitness = GAAlgorithm.evaluate(ind)

            population = offspring

            currentBest = max(population, key=lambda ind: ind.fitness)
            if bestChromosome is None or currentBest.fitness > bestChromosome.fitness:
                bestChromosome = currentBest

            fits = [ind.fitness for ind in population]
            avg = np.mean(fits)
            std = np.std(fits)
            result += f"Gen {gen}: Avg={avg:.4f}, Std={std:.4f}, Min={min(fits):.4f}, Max={max(fits):.4f}\n"

        selectedFeaturesIndices = [i for i, g in enumerate(bestChromosome.genes) if g == 1]

        result += "\n--- GA Results ---\n"
        result += f"Best Chromosome: {bestChromosome.genes}\n"
        result += f"Number of selected features: {len(selectedFeaturesIndices)}\n"
        result += f"Indices of selected features: {selectedFeaturesIndices}\n"
        result += f"Fitness of best solution: {bestChromosome.fitness:.4f}\n"

        return result



