import os
import re
import numpy as np
import matplotlib.pyplot as plt

def parse_page_links():
    pages_dir = "Pages"
    pages_links = []

    if not os.path.exists(pages_dir):
        raise FileNotFoundError(f"Directory '{pages_dir}' does not exist!")

    pattern = r"link:to:(\w+)"

    for filename in os.listdir(pages_dir):
        file_path = os.path.join(pages_dir, filename)
        if filename.endswith(".html"):
            with open(file_path, 'r') as file:
                pages_links.append(re.findall(pattern, file.read()))

    return pages_links

def construct_link_matrix():
    page_links = parse_page_links()
    pages_count = len(page_links)
    link_matrix = np.zeros((pages_count, pages_count))

    for i, link_targets in enumerate(page_links):
        if link_targets:
            unique_links, counts = np.unique(link_targets, return_counts=True)
            for link, count in zip(unique_links, counts):
                j = int(link.replace('Page', '')) - 1
                link_matrix[j, i] = count / len(link_targets)
        else:
            link_matrix[:, i] = 1 / pages_count

    return link_matrix

def solve_eigen_problem(A, num_iterations=10000, tolerance=1e-10):
    r = np.ones(A.shape[0])

    for _ in range(num_iterations):
        new_r = A @ r / np.sum(A, axis=1)
        if np.linalg.norm(new_r - r) < tolerance:
            break
        r = new_r

    return r / np.sum(r)

def power_method(matrix, num_iterations=10000, tolerance=1e-10):
    eigenvector = np.ones(matrix.shape[0]) / matrix.shape[0]

    for _ in range(num_iterations):
        next_vector = matrix @ eigenvector
        next_vector /= np.sum(next_vector)

        if np.linalg.norm(next_vector - eigenvector) < tolerance:
            break

        eigenvector = next_vector

    return eigenvector

def plot_results(pages, eigenvector, rank_positions):
    width = 0.2
    x = np.arange(len(pages))

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.bar(x - width/2, eigenvector, width, label='PageRank Scores', color='olive')
    ax.bar(x + width/2, rank_positions, width, label='Rank Positions', color='royalblue')

    ax.set_xlabel('Pages')
    ax.set_ylabel('Values')
    ax.set_title('Comparison of PageRank Scores and Rank Positions for Pages')
    ax.set_xticks(x)
    ax.set_xticklabels(pages, rotation=45, ha="right")
    ax.legend()

    plt.tight_layout()
    plt.show()

def parse_page_weighted_links():
    pages_dir = "WeightedPages"
    pages_links = []

    if not os.path.exists(pages_dir):
        raise FileNotFoundError(f"Directory '{pages_dir}' does not exist!")

    pattern = r"link:to:(\w+):([\d\.]+)"

    for filename in os.listdir(pages_dir):
        file_path = os.path.join(pages_dir, filename)
        if filename.endswith(".html"):
            with open(file_path, 'r') as file:
                links = re.findall(pattern, file.read())
                pages_links.append({link: float(weight) for link, weight in links})

    return pages_links

def construct_weighted_link_matrix():
    page_links = parse_page_weighted_links()
    pages_count = len(page_links)
    link_matrix = np.zeros((pages_count, pages_count))

    for i, link_targets in enumerate(page_links):
        if link_targets:
            total_weight = sum(link_targets.values())
            for link, weight in link_targets.items():
                j = int(link.replace('Page', '')) - 1
                link_matrix[j, i] = weight / total_weight
        else:
            link_matrix[:, i] = 1 / pages_count

    return link_matrix

matrix = construct_link_matrix()
eigenvector1 = power_method(matrix)
eigenvector2 = solve_eigen_problem(matrix)

pages = [f"Page {i+1}" for i in range(len(eigenvector1))]
rank_positions = np.argsort(-eigenvector1) + 1

print("\nPower Method Results:")
for page in np.argsort(-eigenvector1):
    print(pages[page])

print("\nEigen Problem Results:")
for page in np.argsort(-eigenvector2):
    print(pages[page])

plot_results(pages, eigenvector1, rank_positions)

weighted_matrix = construct_weighted_link_matrix()
weighted_eigenvector = power_method(weighted_matrix)
rank_positions_weighted = np.argsort(-weighted_eigenvector) + 1

print("\nWeighted Power Method Results:")
for page in np.argsort(-weighted_eigenvector):
    print(pages[page])

plot_results(pages, weighted_eigenvector, rank_positions_weighted)
