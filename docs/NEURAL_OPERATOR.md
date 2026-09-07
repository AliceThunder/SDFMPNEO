# Physics Neural Evolution Operator

The extended model combines analytical thermal decay with a trainable physical correction:

```
a(t)=a0 exp(-Lambda t)+Delta a_theta(a0,U,t,G)
```

The network is optimized without solution data. The loss is generated from the electrothermal governing residual:

```
R = da/dt + Lambda a - g_em(a,U,G)
```

Geometry enters through a deterministic physical encoder. The model predicts thermal evolution and coupled WPT quantities including impedance, loss and temperature.
