# tests/unit/test_plot.py
import pytest
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pydas.plot import plot_channel

def test_plot_channel_matplotlib(pydas_instance):
    """Test plotting with matplotlib backend."""
    pydas_instance.__lam__ = 1.0
    fig = plot_channel(pydas_instance, ch_name='Wave1', plotbackend='matplotlib', show=False)
    
    assert fig is not None
    assert isinstance(fig, plt.Figure)

def test_plot_channel_plotly(pydas_instance):
    """Test plotting with plotly backend."""
    pydas_instance.__lam__ = 1.0
    fig = plot_channel(pydas_instance, ch_name='Wave1', plotbackend='plotly', show=False)
    
    assert fig is not None
    assert isinstance(fig, go.Figure) or type(fig).__name__ == 'FigureResampler'
