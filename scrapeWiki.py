import requests
import networkx as nx
import os

def fetch_wikipedia_page_links(page_title):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": page_title,
        "prop": "links",
        "pllimit": "max",
        "plnamespace": "0",
        "format": "json"
    }
    response = requests.get(url, params=params)
    data = response.json()
    pages = data.get("query", {}).get("pages", {})
    links = []
    for page in pages.values():
        if "links" in page:
            links.extend([link["title"] for link in page["links"]])
    return links

def create_wikipedia_graph(seed_page, num_nodes):
    G = nx.DiGraph()
    queue = [seed_page]
    visited = set()

    while queue and len(G) < num_nodes:
        current_page = queue.pop(0)
        #print(current_page)
        if current_page in visited:
            continue
        visited.add(current_page)
        links = fetch_wikipedia_page_links(current_page)
        for link in links:
            G.add_edge(current_page, link)
            if link not in visited and len(G) < num_nodes:
                queue.append(link)
        print(f"Processed {len(visited)} pages, queue size: {len(queue)}")
    print("\n\n\n")
    total = G.nodes
    i = 0
    for pages in G.nodes:
        i = i + 1
        print(i, " out of ", len(total))
        links = fetch_wikipedia_page_links(pages)
        for link in links:
            if not G.has_edge(pages, link) and link in G.nodes:
                G.add_edge(pages, link)
    return G


seed_page = "Machine learning"
num_nodes = 1000
wikipedia_graph = create_wikipedia_graph(seed_page, num_nodes)

# Save the graph to a file
nx.write_gml(wikipedia_graph, "wikipedia_subset_small.gml")

print(f"Graph created with {len(wikipedia_graph.nodes)} nodes and {len(wikipedia_graph.edges)} edges")