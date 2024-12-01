use std::collections::HashMap;

use crate::{Impossible, Label, UnlinkedTextSectionElement};

use eyre::Result;
use log::debug;

pub fn link(mut code: Vec<Label>) -> Result<Vec<u8>> {
    let main = code.remove(
        code.iter()
            .position(|label| label.ident.as_str() == "main")
            .impossible()?,
    );

    code.insert(0, main);

    let mut function_map = HashMap::new();

    let mut position = 0;
    for label in &code {
        function_map.insert(label.ident.clone(), position);

        for code in &label.code {
            match code {
                UnlinkedTextSectionElement::Binary(binary) => position += binary.len(),
                UnlinkedTextSectionElement::Call { function_name: _ } => position += 5,
            }
        }
    }

    let mut result = Vec::new();
    for label in code {
        for code in label.code {
            match code {
                UnlinkedTextSectionElement::Binary(mut binary) => result.append(&mut binary),
                UnlinkedTextSectionElement::Call { ref function_name } => {
                    let Some(address) = function_map.get(function_name) else {
                        panic!("`{function_name}` not in linker hashmap.");
                    };
                    result.push(0xE8);
                    let result_address = *address as i32 - result.len() as i32 - 5;
                    debug!("Call for `{function_name}` from {} to {address} by jumping {result_address}", result.len());
                    result.extend_from_slice(&mut result_address.to_le_bytes());
                }
            }
        }
    }

    Ok(result)
}
