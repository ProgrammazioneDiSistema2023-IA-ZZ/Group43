use onnx_runner::parser::parser::Parser;
use onnx_runner::onnx::onnx_graph::OnnxGraph;

fn main() {
    println!("Welcome to the onnx runner!");
    //test_parser();
    test_onnx();
    // test_tmp();
}

fn test_parser() {
    println!("Test parser");
    let onnx_file = "googlenet-12";
    let onnx_model = Parser::extract_from_onnx_file(onnx_file).expect("Error in onnx file parsing");
    Parser::store_to_json_file(onnx_file, & onnx_model).expect("Error in storing json");
    let _onnx_model_from_json = Parser::extract_from_json_file(onnx_file).expect("Error in json file parsing");
}

fn test_onnx() {
    let onnx_file = "mnist-12";
    let onnx_model = Parser::extract_from_json_file(onnx_file).expect("Error in json file parsing");
    let onnx = OnnxGraph::from(onnx_model);
    println!("{}", onnx);
}

fn test_tmp(){
    let onnx_file = "googlenet-12";
    let onnx_model = Parser::extract_from_json_file(onnx_file).expect("Error in json file parsing");
    let result = onnx_model.graph.node.iter().all(|n| n.output.len() == 1);
    if result{
        println!("PASSED!");
    }else {
        println!("FAILED!");
    }
}