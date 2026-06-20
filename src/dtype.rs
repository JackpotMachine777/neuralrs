#[derive(Clone, Debug, PartialEq)]
/// The element type of a tensor's values. Currently just 32-bit float.
pub enum DType{
    Float32,
    Float64,
    Int32,
    Int64,
    Bool,
}