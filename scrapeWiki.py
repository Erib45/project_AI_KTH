import aiohttp
import asyncio
import networkx as nx
import time

async def fetch_wikipedia_page_links(session, page_title, semaphore, max_retries=5):
    async with semaphore:
        for attempt in range(max_retries):
            # Base delay to be polite
            await asyncio.sleep(1) 
            
            url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "titles": page_title,
                "prop": "links",
                "pllimit": "max",
                "plnamespace": "0",
                "format": "json"
            }
            
            headers = {
                "User-Agent": "WikiAI_Pathfinder_Bot/1.0 (mailto:PUT_YOUR_REAL_EMAIL_HERE@gmail.com)",
                "Accept": "application/json"
            }
            
            try:
                async with session.get(url, params=params, headers=headers) as response:
                    
                    # 🚨 THE NEW 429 HANDLING 🚨
                    if response.status == 429:
                        # Exponential backoff: 2s, 4s, 8s, 16s...
                        wait_time = 3 ** attempt
                        print(f"⏳ Rate limited on '{page_title}'. Pausing for {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue # Skip to the next loop iteration to try again!
                        
                    elif response.status != 200:
                        print(f"⚠️ Blocked! Status {response.status} on '{page_title}'.")
                        return page_title, []
                        
                    data = await response.json()
                    pages = data.get("query", {}).get("pages", {})
                    links = []
                    for page in pages.values():
                        if "links" in page:
                            links.extend([link["title"] for link in page["links"]])
                    return page_title, links
                    
            except aiohttp.ContentTypeError:
                print(f"JSON Decode Error on {page_title}. Retrying...")
                await asyncio.sleep(2)
                continue
            except Exception as e:
                print(f"Error fetching {page_title}: {e}")
                return page_title, []
                
        # If we loop 5 times and fail every time, finally give up.
        print(f"❌ Failed to fetch '{page_title}' after {max_retries} attempts.")
        return page_title, []

async def create_wikipedia_graph(seed_page, num_nodes):
    G = nx.DiGraph()
    G.add_node(seed_page)
    
    queue = [seed_page]
    visited_for_links = set() # Track pages we've already fetched links for
    all_fetched_links = {}    # Store links in memory: {page: [link1, link2]}
    
    # Allow 50 concurrent requests to Wikipedia
    semaphore = asyncio.Semaphore(10) 
    
    async with aiohttp.ClientSession() as session:
        print("Phase 1: Expanding the graph nodes...")
        while queue and len(G.nodes) < num_nodes:
            # Take up to 50 items from the queue to process concurrently
            batch = queue[:50]
            queue = queue[50:]
            
            # Create asynchronous tasks for the batch
            tasks = [fetch_wikipedia_page_links(session, page, semaphore) for page in batch if page not in visited_for_links]
            
            if not tasks:
                continue
                
            # Run the batch concurrently
            results = await asyncio.gather(*tasks)
            
            for page_title, links in results:
                visited_for_links.add(page_title)
                all_fetched_links[page_title] = links
                
                for link in links:
                    if len(G.nodes) < num_nodes:
                        if not G.has_node(link):
                            G.add_node(link)
                            queue.append(link)
                    else:
                        break # Stop adding new nodes once we hit our limit
            
            print(f"Nodes collected: {len(G.nodes)}/{num_nodes} | Queue size: {len(queue)}")

        print("\nPhase 2: Fetching links for remaining leaf nodes...")
        # Our graph has 1000 nodes, but we haven't fetched the outbound links for the last ones added (the leaves)
        remaining_nodes = [node for node in G.nodes if node not in visited_for_links]
        
        # Batch fetch the remaining nodes concurrently
        tasks = [fetch_wikipedia_page_links(session, page, semaphore) for page in remaining_nodes]
        # Chunk them to avoid overwhelming the system
        chunk_size = 100
        for i in range(0, len(tasks), chunk_size):
            chunk_results = await asyncio.gather(*tasks[i:i+chunk_size])
            for page_title, links in chunk_results:
                all_fetched_links[page_title] = links
            print(f"Fetched leaf links: {min(i+chunk_size, len(tasks))}/{len(tasks)}")

    print("\nPhase 3: Building all local edges (No API calls!)...")
    # Now that we have all the links for our 1000 nodes saved in memory, 
    # we just connect the dots locally.
    nodes_set = set(G.nodes)
    for source_page, target_links in all_fetched_links.items():
        if source_page in nodes_set:
            for target_page in target_links:
                if target_page in nodes_set:
                    G.add_edge(source_page, target_page)

    return G

async def main():
    start_time = time.time()
    seed_page = "Machine learning"
    num_nodes = 10_000
    
    wikipedia_graph = await create_wikipedia_graph(seed_page, num_nodes)

    # Save the graph to a file
    nx.write_gml(wikipedia_graph, "wikipedia_subset_small.gml")

    end_time = time.time()
    print(f"\nGraph created with {len(wikipedia_graph.nodes)} nodes and {len(wikipedia_graph.edges)} edges.")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    # Windows specific fix for asyncio
    import sys
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    asyncio.run(main())