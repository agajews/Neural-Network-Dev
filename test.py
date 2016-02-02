def get_conv_output_length(input_length, filter_size, stride,
                           pad):
    if not pad:
        output_length = input_length - filter_size + 1
    else:
        output_length = input_length + filter_size - 1
    return (output_length + stride - 1) // stride
dim = get_conv_output_length(1000, 10, 2, False)
dim = dim ** 2
dim2 = get_conv_output_length(1000, 20, 2, False)
dim2 = dim2 ** 2
print(dim + dim2)
