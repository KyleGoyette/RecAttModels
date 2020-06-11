"""
plot_grads_attn
{
    "$schema": "https://vega.github.io/schema/vega-lite/v4.json",
    "padding": 5,
    "width": 500,
    "height": 500,
    "data": {"name": "${history-table:rows:x-axis,key}"},
    "title": {
      "text": {"value": "Gradient Propagation Through Time"}
    },
    "layer":
      [
        {
        "data": "grads"
          "selection":
          {
            "single":
            {
              "type": "single",
              "on": "mouseover",
              "nearest": true,
              "fields": ["step"],
              "empty": "none"
            }
          }
          "mark": {"type": "line"},
          "encoding":
          {
            "x": {"field": "step", "type": "quantitative", "axis": {"title": "timestep"}},
            "y": {"field": "grad", "type": quantitative, "axis": {"title": "|dL/dh|"}},
            "opacity": {"value": 0.7}
          }
        },
        {
          "mark": {"type": "line"},
          "encoding":
          {
            "x1": {"field": "origin_x", "type": "quantitative"},
            "y1": {"field": "origin_y", "type": "quantitative"},
            "x2": {"field": "destination_x", "type": "quantitative"},
            "y2": {"field": "destination_y", "type": "quantitative"},
            "opacity": {"field": "attn", "type": "quantitative"}
          },
          "transform":
          [
            {"filter": "{"selection": "single"}}
          ]
        }
      ]
    }
"""