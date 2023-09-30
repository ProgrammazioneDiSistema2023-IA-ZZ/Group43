use std::fs;
use std::fs::File;
use std::string::String;
use protobuf::{CodedInputStream, Message};
use crate::parser::onnx_model::onnx_proto3::ModelProto;

const ONNX_EXTENSION: &str = ".onnx";
const JSON_EXTENSION: &str = ".json";

pub struct Parser{}

impl Parser{
    pub fn extract_from_onnx_file(file_name: &str) -> Result<ModelProto, Box<dyn std::error::Error>>{
        let f_name = if file_name.contains(ONNX_EXTENSION){
            String::from(file_name)
        } else {
            String::from(file_name) + ONNX_EXTENSION
        };
        let mut buffer = File::open(f_name)?;
        let mut in_stream = CodedInputStream::new(&mut buffer);
        let mut onnx_model_proto = ModelProto::new();
        onnx_model_proto.merge_from(&mut in_stream)?;
        Ok(onnx_model_proto)
    }

    pub fn extract_from_json_file(file_name: &str) -> Result<ModelProto, Box<dyn std::error::Error>>{
        let f_name = if file_name.contains(JSON_EXTENSION){
            String::from(file_name)
        } else {
            String::from(file_name) + JSON_EXTENSION
        };
        let json = fs::read_to_string(f_name)?;
        Self::extract_from_json(&json)
    }

    fn extract_from_json(json: &String) -> Result<ModelProto, Box<dyn std::error::Error>>{
        let mut onnx_model_proto = ModelProto::new();
        protobuf_json_mapping::merge_from_str(&mut onnx_model_proto, json.as_str())?;
        Ok(onnx_model_proto)
    }

    pub fn store_to_json_file(file_name: &str, onnx: &ModelProto) -> Result<(), Box<dyn std::error::Error>>{
        let f_name = if file_name.contains(JSON_EXTENSION){
            String::from(file_name)
        } else {
            String::from(file_name) + JSON_EXTENSION
        };
        let json = protobuf_json_mapping::print_to_string(onnx)?;
        fs::write(f_name, json.as_str())?;
        Ok(())
    }
}