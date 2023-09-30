use onnx_runner::parser::parser::Parser;

fn main() {
    println!("Welcome to the onnx runner!");
    let onnx_file = "mnist-12";
    let onnx_model = Parser::extract_from_onnx_file(onnx_file).expect("Error in onnx file parsing");
    Parser::store_to_json_file(onnx_file, & onnx_model).expect("Error in storing json");
    let onnx_model_from_json = Parser::extract_from_json_file(onnx_file).expect("Error in json file parsing");
}
