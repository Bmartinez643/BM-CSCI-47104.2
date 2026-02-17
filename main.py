import spacy
import networkx as nx
import matplotlib.pyplot as plt

nlp = spacy.load("en_core_web_trf")


def normalize_verb(token):
    communication_verbs = {
        "say", "tell", "ask", "reply", "cry", "shout",
        "whisper", "scream", "exclaim", "remark",
        "mutter", "answer", "call"
    }

    lemma = token.lemma_.lower()

    if lemma in communication_verbs:
        return "say"
    return lemma


# Define the function for simple Semantic Role Labeling (SRL)
def simple_srl(sentence):
    subjects = []
    verbs = []
    objects = []
    indirect_objects = []

    for token in sentence:
        if "subj" in token.dep_:
            subjects.append(token.text)
        if "VERB" in token.pos_:
            verbs.append(normalize_verb(token))
        if "obj" in token.dep_:
            objects.append(token.text)
        if "dative" in token.dep_:
            indirect_objects.append(token.text)

    return {
        'subjects': subjects,
        'verbs': verbs,
        'objects': objects,
        'indirect_objects': indirect_objects
    }


def build_and_plot_knowledge_graph_matplotlib(srl_results):
    G = nx.DiGraph()

    for result in srl_results:
        subjects = result['subjects']
        verbs = result['verbs']
        objects = result['objects']
        indirect_objects = result['indirect_objects']

        for subject in subjects:
            for verb in verbs:
                for obj in objects:
                    G.add_edge(subject, obj, label=verb)
                for ind_obj in indirect_objects:
                    G.add_edge(subject, ind_obj, label=verb)

    pos = nx.spring_layout(G, seed=42)

    # Draw nodes and edges
    nx.draw(G, pos, with_labels=True, node_color="skyblue", node_size=2000,
            font_size=12, font_color="black",
            font_weight="bold", arrows=True)

    # Draw edge labels
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # Show plot
    plt.show()


def load_text(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    return text


text = load_text("alice.txt")

# Process each sentence and extract SRL results
srl_results = []
doc = nlp(text)
for sent in doc.sents:
    result = simple_srl(sent)
    srl_results.append(result)

# Build and plot the knowledge graph with matplotlib
# build_and_plot_knowledge_graph_matplotlib(srl_results)


def query_graph(srl_results, subject_node, verb_edge):
    G = nx.DiGraph()

    for result in srl_results:
        subjects = result['subjects']
        verbs = result['verbs']
        objects = result['objects']
        indirect_objects = result['indirect_objects']

        for subject in subjects:
            for verb in verbs:
                for obj in objects:
                    G.add_edge(subject, obj, label=verb)
                for ind_obj in indirect_objects:
                    G.add_edge(subject, ind_obj, label=verb)

    answer = []
    edges = G.out_edges(subject_node)
    for u, v in edges:
        if G[u][v].get("label") == verb_edge:
            answer.append(v)
    return answer



def is_action_verb(token):
    if token.pos_ == "VERB":
        return normalize_verb(token)
    return None


query = "Where did Alice go?"
for token in nlp(query):
    result = is_action_verb(token)
    if result == "go":
        print(query_graph(srl_results, "Alice", result))
        break


query = "What did Alice say?"
for token in nlp(query):
    result = is_action_verb(token)
    if result == "say":   # ✅ fixed bug here
        print(query_graph(srl_results, "Alice", result))
        break