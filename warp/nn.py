import warp

class Linear:

    def __init__(self, dim_in, dim_out):

        # initialize layer data
        self.weights = warp.zeros((dim_in, dim_out), dtype=float)
        self.bias = warp.zeros(dim_out, dtype=float)

    def finalize(self, name, module):

        class LinearDesc:

            weights: warp.array2d(dtype=float)
            bias: warp.array1d(dtype=float)

        # construct layer type and register to owning module
        layer_type = warp.codegen.Struct(cls=LinearDesc, key=f"_wp_nn_layer_desc_{name}", module=module)
        
        # construct layer args
        self.__struct__ = layer_type()
        self.__struct__.weights = self.weights
        self.__struct__.bias = self.bias

        dim_in = self.weights.shape[0]
        dim_out = self.weights.shape[1]

        # function to perform forward evaluation inside the kernel
        def eval(layer: layer_type, x: warp.types.vector(length=dim_in, dtype=float)):
            
            y = warp.vector(length=dim_out, dtype=float)

            for i in range(dim_out):
                for j in range(dim_in):
                    y[i] += layer.weights[i,j]*x[j] + layer.bias[i]

            return y
        
        layer_func = warp.context.Function(func=eval,
                                           key=f"_wp_nn_layer_desc_eval_{name}",
                                           namespace="",
                                           module=module,
                                           value_func=None)

        return layer_type, layer_func
        

class Module:

    layers = []

    def add(self, name, layer):
        self.layers.append((name, layer))

    # construct struct type, flatten, make forward funcs visible
    def compile(self):
        
        module = warp.context.get_module(self.__module__)

        class ModuleDesc:
            pass

        # we will dynamically build annoations
        # for layers to later construct the Warp struct from
        ModuleDesc.__annotations__ = {}
       
        for (name, layer) in self.layers:

            # create layer struct, args, and eval function
            layer_type, layer_func = layer.finalize(name, module)

            # setup struct description for each layer
            ModuleDesc.__annotations__[name] = layer_type
       
        # create struct for module as a whole
        module_type = warp.codegen.Struct(cls=ModuleDesc, key=f"_wp_nn_module_desc_name", module=module)

        # construct new instance
        model = module_type()

        # copy per-layer args into module struct
        for (name, layer) in self.layers:
            setattr(model, name, layer.__struct__)

        return model

        
        

