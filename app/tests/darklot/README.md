SYNTHETIC test photos for the "dark lot" rule (src/lib/darklot.ts), made from `public/demo/tray_synthetic_00.jpg`
(itself a SYNTHETIC composite of J4ckDev bean crops, CC BY-NC-SA 4.0) by recolouring every bean pixel with its
luminance kept as texture (out = L / L90 x RGB):
- `black_beans_synthetic.jpg`: RGB 58,56,44 (olive-black, low saturation, like black green beans) -> expected c07 "not sure"
- `roast_dark_synthetic.jpg`: RGB 62,38,24 (REGRESSION_B's dark-roast recolour) -> expected c06 "not green coffee"
