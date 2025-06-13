"""
Diagram of Various Decoder Architectures (GRU, LSTM, LiGRU, LinearLag)
----------------------------------------------------------------------
Install Graphviz and its Python bindings:
    pip install graphviz

Make sure Graphviz system binaries are also installed:
    - Linux (Ubuntu): sudo apt-get install graphviz
    - macOS: brew install graphviz
    - Windows: https://graphviz.org/download/
"""

from graphviz import Digraph

def create_decoder_architecture_diagram():
    dot = Digraph("DecoderArchitectures", filename="decoders_diagram", format="png")
    # Set some global attributes for a cleaner look
    dot.attr(rankdir='LR', splines='ortho', fontname="Arial", nodesep='1', ranksep='2')

    # ------------------- GRUDecoder Subgraph ------------------- #
    with dot.subgraph(name='cluster_GRU') as c:
        c.attr(label="GRUDecoder", color="blue", style="rounded,dashed", penwidth="2")
        c.node("GRU_in", label="Input\n(batch, seq_len, input_size)", shape="box")
        c.node("GRU_layer", label="GRU\n(hidden_size)", shape="box")
        c.node("GRU_fc", label="Linear\n(hidden_size → output_size)", shape="box")
        c.node("GRU_out", label="Output", shape="box")

        c.edges([("GRU_in","GRU_layer"), ("GRU_layer","GRU_fc"), ("GRU_fc","GRU_out")])

    # ------------------- LSTMDecoder Subgraph ------------------- #
    with dot.subgraph(name='cluster_LSTM') as c:
        c.attr(label="LSTMDecoder", color="red", style="rounded,dashed", penwidth="2")
        c.node("LSTM_in", label="Input\n(batch, seq_len, input_size)", shape="box")
        c.node("LSTM_layer", label="LSTM\n(hidden_size)", shape="box")
        c.node("LSTM_fc", label="Linear\n(hidden_size → output_size)", shape="box")
        c.node("LSTM_out", label="Output", shape="box")

        c.edges([("LSTM_in","LSTM_layer"), ("LSTM_layer","LSTM_fc"), ("LSTM_fc","LSTM_out")])

    # ------------------- LiGRUDecoder Subgraph ------------------- #
    with dot.subgraph(name='cluster_LiGRU') as c:
        c.attr(label="LiGRUDecoder", color="green", style="rounded,dashed", penwidth="2")
        c.node("LiGRU_in", label="Input\n(batch, seq_len, input_size)", shape="box")
        c.node("LiGRU_cell", label="LiGRU Cell\n(hidden_size)", shape="box")
        c.node("LiGRU_fc", label="Linear\n(hidden_size → output_size)", shape="box")
        c.node("LiGRU_out", label="Output", shape="box")

        # We might indicate unrolled steps, but keep it simple here
        c.edges([("LiGRU_in","LiGRU_cell"), ("LiGRU_cell","LiGRU_fc"), ("LiGRU_fc","LiGRU_out")])

    # ------------------- LinearLagDecoder Subgraph ------------------- #
    with dot.subgraph(name='cluster_LinearLag') as c:
        c.attr(label="LinearLagDecoder", color="orange", style="rounded,dashed", penwidth="2")
        c.node("LL_in", label="Input\n(batch, input_dim)", shape="box")
        c.node("LL_linear1", label="Linear\n(input_dim → hidden_dim)", shape="box")
        c.node("LL_relu", label="ReLU", shape="box")
        c.node("LL_linear2", label="Linear\n(hidden_dim → output_dim)", shape="box")
        c.node("LL_out", label="Output", shape="box")

        c.edges([
            ("LL_in","LL_linear1"),
            ("LL_linear1","LL_relu"),
            ("LL_relu","LL_linear2"),
            ("LL_linear2","LL_out")
        ])

    return dot

if __name__ == "__main__":
    diagram = create_decoder_architecture_diagram()
    diagram.render(cleanup=True)  # Renders to 'decoders_diagram.png' by default