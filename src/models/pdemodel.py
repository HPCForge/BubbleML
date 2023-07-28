

def get_model(args, pde):
    if args.name in MODEL_REGISTRY:
        _model = MODEL_REGISTRY[args.name].copy()
        _model["init_args"].update(
            dict(
                n_input_scalar_components=pde.n_scalar_components,
                n_output_scalar_components=pde.n_scalar_components,
                n_input_vector_components=pde.n_vector_components,
                n_output_vector_components=pde.n_vector_components,
                time_history=args.time_history,
                time_future=args.time_future,
                activation=args.activation,
            )
        )
        model = instantiate_class(tuple(), _model)
    else:
        logger.warning("Model not found in registry. Using fallback. Best to add your model to the registry.")
        if hasattr(args, "time_history") and args.model["init_args"]["time_history"] != args.time_history:
            logger.warning(
                f"Model time_history ({args.model['init_args']['time_history']}) does not match data time_history ({pde.time_history})."
            )
        if hasattr(args, "time_future") and args.model["init_args"]["time_future"] != args.time_future:
            logger.warning(
                f"Model time_future ({args.model['init_args']['time_future']}) does not match data time_future ({pde.time_future})."
            )
        model = instantiate_class(tuple(), args.model)

    return model
