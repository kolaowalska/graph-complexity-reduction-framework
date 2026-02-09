# graph complexity reduction framework

## overview & purpose
this project is a panoramic, research-oriented pipeline for evaluating graph reduction algorithms. it's designed to support thesis-level experimentation by allowing the user to compare different algorithms that reduce graph complexity and the ways they influence graph properties in large-scale networks. 

the core goal of this framework is to become a tool for the systematic study of reduction techniques and potentially evolve into a research environment as an automated laboratory for graph theory.

*__note__: the project is an unfolding consequence of my bachelor's `thesis.pdf`. it is still very much a work in progress (hence the demo) but is constantly evolving to hopefully become part of my postgraduate research at jagiellonian university.*

### general functionalities

* __multi-source graph ingestion__: loading graphs from various formats (edgelists, memory objects) via a unified gateway. additionally, the framework uses lazy loading to boost performance while handling massive datasets
* __polymorphic transformations__: support for different types of graph reduction
  - __sparsification__ - selection of significant nodes/edges and discarding others
  - __coarsening__ - aggregation of similar nodes/edges to construct a smaller graph
  - __condensation (work in progress)__ - learning a synthetic graph from scratch
* __automated metric registry__: calculating structural properties in the original and modified graph on the fly 
* __experiment management__: orchestrating full experiments from start to finish, where a graph is imported, transformed, analyzed, and the results are persisted as an audit trail
* __efficiency benchmarking__: automatic tracking of wall-clock time for both transformation and metric phases to evaluate theoretical vs. empirical complexity
* __visualization__: basic metric value plots and graph figures

### system architecture
the framework is built using domain-driven design principles and organized into distinct layers to ensure that experimental logic remains decoupled from infrastructure concerns. the plugin-driven architecture ensures flexibility and makes the program open for future extension.

- `domain` layer contains the core truth of the system, such as the `Graph` model, `Metric` definitions, and the abstract `GraphTransform` logic; it is purely focused on graph theory and the mathematics behind the graph algorithms
- `application` layer orchestrates the workflow via the `ExperimentService`, handling the "business logic" of running a research job without needing to know how graphs are stored and where they come from
- `infrastructure` layer takes care of details like reading graphs (`GraphGateway`), persisting results (`Repository`), and managing database transactions (`UnitofWork`)
- `interface` layer provides entry points for the user including a CLI for automated batches and an API for potential integration with web dashboards

### the pipeline
the program follows a defined lifecycle for every experiment.
1. __import__: the `GraphGateway` reads raw data and creates a lazy object
2. __orchestration__: the `ExperimentService` receives a command via a CLI/API and fetches the original graph from the `Repository`
3. __transformation__: the system looks up the requested algorithm in a `Registry` and executes it, producing a new `Graph` object while keeping the original graph intact
4. __analysis__: a series of `Metrics` is run against the graph and return detailed dictionaries of values and metadata from the calculation
5. __commitment__: the `UnitofWork` ensures that both the new graph and the experiment results are saved to storage simultaneously, preventing dirty data resulting from errors

## quick start

### experimental setup
experiments are managed through the `ExperimentService`, which coordinates the lifecycle of importing, transforming and measuring graphs.

### running the demo
1. clone or download the repository to your local machine
2. open the terminal and navigate to the project's root directory 
3. run the following demo that showcases the pipeline from ingestion to visualization:
~~~python
python src/demo.py
~~~
this script runs a small selection of scenarios and outputs comparative metric results along with some basic graph visualizations.

## extensibility
### algorithm agnosticism
it makes no difference whether the operation is removing edges, merging nodes, or reweighting the entire graph; as long as the algorithm follows the `GraphTransform` interface, it can be plugged into the pipeline without changing a single line of core code.

### dynamic metric discovery 
adding a new metric requires only creating a single new file, and the system automatically discovers new metrics at runtime using a scanning decorator. this makes it trivial to add new evaluation criteria with progressing research needs.

### interchangeable storage
the current system uses an in-memory storage for speed and demo purposes, but the architecture is decoupled such that swapping to a SQL database or a specialized graph database (like Neo4j) requires only changing the `Repository` implementation, while the rest of the logic remains untouched.

### future directions
this project is designed to eventually support adapting reduction strategies in dynamic graphs and automating the selection of reduction techniques based on graph topology for machine learning integration. 
