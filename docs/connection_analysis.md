# Sensor connection information

The supplied 54-sensor heatmap shows pairwise connection probabilities. The grouped analysis summarizes mean probabilities by distance, intervening wall count and area membership. Nearby pairs with fewer intervening walls generally have higher connection probabilities. The released building tables use the annotation exporter's geometry-dependent prior, exp(−2.2d/L − 0.85w), where d is distance, L is the annotated plan diagonal and w is the intervening wall count. BRICK normalizes this prior and uses it to modulate graph edge weights with the fixed coefficient 0.5.

![Pairwise connection probabilities](connection_analysis/connection_probability_heatmap_sensor_54x54.png)

![Connection probability by distance and wall separation](connection_analysis/combined_effect_visualization.png)
