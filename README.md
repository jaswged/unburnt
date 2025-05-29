# Unburnt

AshToCache

. Note that a constructor will automatically be generated for your configuration, which will take in as input values the parameters which do not have default values: let config = ModelConfig::new(num_classes, hidden_size);. The default values can be overridden easily with builder-like methods: (e.g config.with_dropout(0.2);)

Burn provides two basic output types: `ClassificationOutput` and `RegressionOutput`
