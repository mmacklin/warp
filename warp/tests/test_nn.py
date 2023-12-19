import warp as wp
import warp.nn

from typing import Any

wp.init()

# model = wp.nn.Module()
# model.add("layer1", wp.nn.Linear(12, 8))
# model.add("layer2", wp.nn.Linear(8, 4))
# model.finalize()

# m = wp.nn.Module()
# m.add("layer1", None)

# @wp.struct
# class Model:

#     # need to declare types
#     layer1: wp.nn.Linear
#     layer2: wp.nn.Linear

#     def __init__(self):
        
#         # and explicitly initialize them
#         layer1 = wp.nn.Linear(12, 8)
#         layer2 = wp.nn.Linear(8, 4)


#     def kernel():
        
#         x = wp.vector(12, dtype=float)
        
#         y = wp.nn.linear(model.layer1, 8, x)
#         z = wp.nn.linear(model.layer2, 4, y)



class Model(wp.nn.Module):

    def __init__(self):
        
        self.add("layer1", wp.nn.Linear(12, 8))
        self.add("layer2", wp.nn.Linear(8, 4))
        
model = Model().compile()
cls = model._cls

@wp.kernel
def forward(model: cls):

    x = wp.vector(length=8, dtype=float)

    # y = eval(model.layer1, x)
    # z = eval(model.layer2, y)

    print(x)

wp.launch(forward, dim=1, inputs=[model])
wp.synchronize()

# def linear(dim_in, dim_out, bias=True, dtype=float, requires_grad=True):

#     @wp.struct
#     class Linear:
        
#         weights : wp.array2d(dtype=dtype)
#         bias : wp.array1d(dtype=dtype)

#         def __init__(self,
#                      dim_in,
#                      dim_out,
#                      bias=True,
#                      dtype=float,
#                      requires_grad=True):
            
#             self.input = wp.types.vector(length=dim_in, dtype=dtype)
#             self.output = wp.types.vector(length=dim_out, dtype=dtype)
            
#             self.weights = wp.array2d(dim_in,
#                                       dim_out,
#                                       dtype=dtype,
#                                       requires_grad=requires_grad)

#             if bias:
#                 self.bias = wp.array1d(dim_out,
#                                        dtype=dtype,
#                                        requires_grad=requires_grad)

#         @wp.func
#         def eval(x: wp.vector(dtype=dtype)):

            
#             for i in range(dim_out):
#                 for j in range(dim_in):




# @wp.struct
# class Module:

#     def __init__(self):
        
#         self.linear1 = wp.nn.Linear(16, 32)
#         self.linear2 = wp.nn.Linear(32, 4)


# @wp.kernel
# def evaluate(model: Module):

#     x = wp.vector(16, dtype=float)

#     x = wp.map(model.linear1, x)
#     x = wp.map(relu, x)

#     x = wp.map(model.linear2, x)
#     x = wp.map(relu, x)