use std::collections::{HashMap, VecDeque};

use crate::{Impossible, Label, UnlinkedTextSectionElement};

use eyre::Result;
use log::debug;

pub fn link(code: Vec<Label>) -> Result<Vec<u8>> {
    let mut res = Vec::new();
    let mut code = VecDeque::from(code);

    while let Some(next) = code.pop_front() {
        let ident = next.ident.to_string();
        let mut this_vec = vec![next.clone()];
        while let Some(position) = code.iter().position(|label| label.ident == ident) {
            let element = code.remove(position).impossible().unwrap();
            this_vec.push(element);
        }
        res.push(this_vec);
    }

    let mut code: Vec<Label> = res
        .into_iter()
        .map(|labels| Label {
            ident: labels[0].ident.to_string(),
            code: labels
                .into_iter()
                .flat_map(|label| label.code.clone())
                .collect(),
        })
        .collect();

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
                    let result_address = *address as i32 - result.len() as i32 - 4;
                    debug!("Call for `{function_name}` from {} to {address} by jumping {result_address}", result.len());
                    result.extend_from_slice(&mut result_address.to_le_bytes());
                }
            }
        }
    }

    Ok(result)
}
