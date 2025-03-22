#include "basic_node_base.h"
NODE_DEF_OPEN_SCOPE
NODE_DECLARATION_FUNCTION(value_add)
{
    b.add_input<float>("A");
    b.add_input<float>("B");
    b.add_output<float>("Result");
}

NODE_EXECUTION_FUNCTION(value_add)
{
    auto a = params.get_input<float>("A");
    auto b = params.get_input<float>("B");
    params.set_output<float>("Result", a + b);
    return true;
}

NODE_DECLARATION_FUNCTION(value_sub)
{
    b.add_input<float>("A");
    b.add_input<float>("B");
    b.add_output<float>("Result");
}

NODE_EXECUTION_FUNCTION(value_sub)
{
    auto a = params.get_input<float>("A");
    auto b = params.get_input<float>("B");
    params.set_output<float>("Result", a - b);
    return true;
}

NODE_DECLARATION_FUNCTION(value_mul)
{
    b.add_input<float>("A");
    b.add_input<float>("B");
    b.add_output<float>("Result");
}

NODE_EXECUTION_FUNCTION(value_mul)
{
    auto a = params.get_input<float>("A");
    auto b = params.get_input<float>("B");
    params.set_output<float>("Result", a * b);
    return true;
}

NODE_DECLARATION_FUNCTION(value_div)
{
    b.add_input<float>("A");
    b.add_input<float>("B");
    b.add_output<float>("Result");
}

NODE_EXECUTION_FUNCTION(value_div)
{
    auto a = params.get_input<float>("A");
    auto b = params.get_input<float>("B");
    params.set_output<float>("Result", a / b);
    return true;
}

NODE_DECLARATION_FUNCTION(value_and)
{
    b.add_input<bool>("A");
    b.add_input<bool>("B");
    b.add_output<bool>("Result");
}

NODE_EXECUTION_FUNCTION(value_and)
{
    auto a = params.get_input<bool>("A");
    auto b = params.get_input<bool>("B");
    params.set_output<bool>("Result", a && b);
    return true;
}
NODE_DECLARATION_FUNCTION(value_or)
{
    b.add_input<bool>("A");
    b.add_input<bool>("B");
    b.add_output<bool>("Result");
}

NODE_EXECUTION_FUNCTION(value_or)
{
    auto a = params.get_input<bool>("A");
    auto b = params.get_input<bool>("B");
    params.set_output<bool>("Result", a || b);
    return true;
}

NODE_DECLARATION_FUNCTION(value_not)
{
    b.add_input<bool>("A");
    b.add_output<bool>("Result");
}

NODE_EXECUTION_FUNCTION(value_not)
{
    auto a = params.get_input<bool>("A");
    params.set_output<bool>("Result", !a);
    return true;
}

NODE_DEF_CLOSE_SCOPE
