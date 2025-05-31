def get_model(model_name, args):
    name = model_name.lower()
    if name == 'create':
        from models.create import create
        return create(args)
    else:
        assert 0
