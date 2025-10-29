import random
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
import time

# ---------- كروموسوم ----------
class Chromosome:
    def __init__(self, gene_size):
        self.genes = [0] * gene_size
        self.fitness = 0.0

    def copy(self):
        new_chrom = Chromosome(len(self.genes))
        new_chrom.genes = self.genes.copy()
        new_chrom.fitness = self.fitness
        return new_chrom

# ---------- خوارزمية GA ----------
class GAAlgorithm:
    N_SAMPLES = 0
    N_FEATURES = 0
    random_gen = random.Random()

    @staticmethod
    def evaluate(chromosome, X, y, cat_features):
        selected_indices = [i for i, g in enumerate(chromosome.genes) if g == 1]
        num_selected = len(selected_indices)
        if num_selected == 0:
            return 0.0, 0.0

        X_selected = X[:, selected_indices]

        # تصحيح فهارس الأعمدة النصية للنسخة المختارة
        selected_cat_features = [selected_indices.index(i) for i in cat_features if i in selected_indices]

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        accuracies = []
        for train_idx, test_idx in cv.split(X_selected, y):
            X_train, X_test = X_selected[train_idx], X_selected[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            model = CatBoostClassifier(verbose=0, iterations=200, random_state=42)
            train_pool = Pool(X_train, y_train, cat_features=selected_cat_features)
            test_pool = Pool(X_test, y_test, cat_features=selected_cat_features)
            model.fit(train_pool)
            acc = model.score(test_pool, y_test)
            accuracies.append(acc)

        accuracy = np.mean(accuracies)
        fitness = accuracy - 0.01 * num_selected
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
    def GAOptimize(X, y, cat_features, n_samples=None, n_features=None):
        GAAlgorithm.N_SAMPLES = n_samples or X.shape[0]
        GAAlgorithm.N_FEATURES = n_features or X.shape[1]

        POPULATION_SIZE = 20
        NGEN = 15
        CXPB = 0.5
        TOURN_SIZE = 3

        start_time = time.time()
        population = GAAlgorithm.init_population(POPULATION_SIZE)
        for ind in population:
            ind.fitness, _ = GAAlgorithm.evaluate(ind, X.values, y, cat_features)

        bestChromosome = None
        for gen in range(NGEN):
            offspring = [GAAlgorithm.tournament_selection(population, TOURN_SIZE) for _ in range(POPULATION_SIZE)]
            for i in range(0, POPULATION_SIZE, 2):
                if GAAlgorithm.random_gen.random() < CXPB:
                    c1, c2 = GAAlgorithm.crossover_two_point(offspring[i], offspring[i + 1])
                    offspring[i], offspring[i + 1] = c1, c2
            for mutant in offspring:
                GAAlgorithm.mutate_flip_bit(mutant, 0.1)
                mutant.fitness, _ = GAAlgorithm.evaluate(mutant, X.values, y, cat_features)

            population = offspring
            currentBest = max(population, key=lambda ind: ind.fitness)
            if bestChromosome is None or currentBest.fitness > bestChromosome.fitness:
                bestChromosome = currentBest

        selectedFeaturesIndices = [i for i, g in enumerate(bestChromosome.genes) if g == 1]
        _, best_accuracy = GAAlgorithm.evaluate(bestChromosome, X.values, y, cat_features)
        end_time = time.time()
        elapsed_time = end_time - start_time

        return {
            "best_chromosome": bestChromosome.genes,
            "selected_features_indices": selectedFeaturesIndices,
            "num_selected_features": len(selectedFeaturesIndices),
            "fitness": float(bestChromosome.fitness),
            "accuracy": float(best_accuracy),
            "elapsed_time_seconds": elapsed_time
        }
