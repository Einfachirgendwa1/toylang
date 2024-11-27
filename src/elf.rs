use crate::Program;

use std::{ops::AddAssign, slice};

const PAGE_SIZE: u64 = 0x1000;
const SECTION_ALIGNMENT: u64 = PAGE_SIZE;

struct Aligned {
    required_padding: u64,
    padding: Vec<u8>,
}

fn align(number: u64, alignment: u64) -> Aligned {
    let required_padding = alignment - number % alignment;
    Aligned {
        required_padding,
        padding: vec![0; required_padding as usize],
    }
}

fn align_vector(vec: &mut Vec<u8>, alignment: u64) -> u64 {
    vec.append(&mut align(vec.len() as u64, alignment).padding);
    vec.len() as u64
}

fn align_program_header_offset(p_vaddr: u64, offset: u64) -> Aligned {
    let required =
        dbg!(PAGE_SIZE + dbg!(p_vaddr % PAGE_SIZE) - dbg!(offset % PAGE_SIZE)) % PAGE_SIZE;
    Aligned {
        required_padding: required,
        padding: vec![0; required as usize],
    }
}

fn inc<X>(x: &mut X) -> X
where
    X: Clone,
    X: AddAssign<X>,
    i8: Into<X>,
{
    let pre = x.clone();
    *x += 1.into();
    pre
}

fn growing_subslice<'a, T, A, F>(vec: &'a [T], f: F) -> impl FnMut() -> A + 'a
where
    T: 'a,
    F: Fn(&'a [T]) -> A + 'a,
{
    let mut n = 0;
    move || f(&vec[0..inc(&mut n) as usize])
}

pub fn extend<A: Clone, B>(vec: &mut Vec<A>, b: B) {
    unsafe {
        vec.extend_from_slice(slice::from_raw_parts(
            &b as *const B as *const A,
            size_of::<B>(),
        ))
    }
}

impl Program {
    pub fn generate_elf(mut self) -> Vec<u8> {
        let mut p_vaddr = 0x400000;

        let mut vec = Vec::new();

        let bin = |slice: &[&str]| {
            slice
                .iter()
                .map(|x| x.as_bytes().to_vec())
                .flatten()
                .collect::<Vec<u8>>()
        };

        let sections = [".text\0", ".data\0", ".shstrtab\0"];
        let mut sections_bin = bin(&sections);
        let mut next_section = growing_subslice(&sections, |slice| bin(slice).len() as u32);

        let text_len = align_vector(&mut self.text, SECTION_ALIGNMENT);
        let data_len = align_vector(&mut self.data, SECTION_ALIGNMENT);

        let elf_header_size = size_of::<Elf64Header>() as u64;
        let elf_program_header_size = size_of::<Elf64ProgramHeader>() as u64;
        let elf_section_header_size = size_of::<Elf64SectionHeader>() as u64;

        let headers = elf_header_size + 2 * elf_program_header_size + 3 * elf_section_header_size;
        let mut header_padding = align(headers, PAGE_SIZE);
        let header_len = headers + header_padding.required_padding;

        let elf_header = Elf64Header {
            e_ident: [0x7F, b'E', b'L', b'F', 2, 1, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0],
            e_type: 2,
            e_machine: 0x3E,
            e_version: 1,
            e_entry: p_vaddr,
            e_phoff: elf_header_size,
            e_shoff: elf_header_size + 2 * elf_program_header_size,
            e_flags: 0,
            e_ehsize: elf_header_size as u16,
            e_phentsize: elf_program_header_size as u16,
            e_phnum: 2,
            e_shentsize: elf_section_header_size as u16,
            e_shnum: 3,
            e_shstrndx: 2,
        };

        let text_program_header = Elf64ProgramHeader {
            p_type: 1,
            p_flags: 1 | 4,
            p_offset: header_len,
            p_vaddr,
            p_paddr: p_vaddr,
            p_filesz: text_len,
            p_memsz: text_len,
            p_align: PAGE_SIZE,
        };
        dbg!(&text_program_header).assert_valid();
        p_vaddr += text_len;

        let data_program_header = Elf64ProgramHeader {
            p_type: 1,
            p_flags: 2 | 4,
            p_offset: header_len + text_len,
            p_vaddr,
            p_paddr: p_vaddr,
            p_filesz: data_len,
            p_memsz: data_len,
            p_align: PAGE_SIZE,
        };
        dbg!(&data_program_header).assert_valid();

        let text_section_header = Elf64SectionHeader {
            sh_name: next_section(),
            sh_type: 1,
            sh_flags: 2 | 4,
            sh_addr: 0x400000,
            sh_offset: header_len,
            sh_size: text_len,
            sh_link: 0,
            sh_info: 0,
            sh_addralign: SECTION_ALIGNMENT,
            sh_entsize: 0,
        };
        let data_section_header = Elf64SectionHeader {
            sh_name: next_section(),
            sh_type: 1,
            sh_flags: 1 | 2,
            sh_addr: 0x400000 + text_section_header.sh_size,
            sh_offset: header_len + text_len,
            sh_size: data_len,
            sh_link: 0,
            sh_info: 0,
            sh_addralign: SECTION_ALIGNMENT,
            sh_entsize: 0,
        };
        let strings_section_header = Elf64SectionHeader {
            sh_name: next_section(),
            sh_type: 3,
            sh_flags: 0,
            sh_addr: 0,
            sh_offset: header_len + text_len + data_len,
            sh_size: sections_bin.len() as u64,
            sh_link: 0,
            sh_info: 0,
            sh_addralign: 0x0,
            sh_entsize: 0,
        };

        extend(&mut vec, elf_header);
        extend(&mut vec, text_program_header);
        extend(&mut vec, data_program_header);
        extend(&mut vec, text_section_header);
        extend(&mut vec, data_section_header);
        extend(&mut vec, strings_section_header);

        vec.append(&mut header_padding.padding);
        vec.append(&mut self.text);
        vec.append(&mut self.data);
        vec.append(&mut sections_bin);

        vec
    }
}

#[repr(C)]
struct Elf64Header {
    e_ident: [u8; 16],
    e_type: u16,
    e_machine: u16,
    e_version: u32,
    e_entry: u64,
    e_phoff: u64,
    e_shoff: u64,
    e_flags: u32,
    e_ehsize: u16,
    e_phentsize: u16,
    e_phnum: u16,
    e_shentsize: u16,
    e_shnum: u16,
    e_shstrndx: u16,
}

#[derive(Debug)]
#[repr(C)]
struct Elf64ProgramHeader {
    p_type: u32,
    p_flags: u32,
    p_offset: u64,
    p_vaddr: u64,
    p_paddr: u64,
    p_filesz: u64,
    p_memsz: u64,
    p_align: u64,
}

impl Elf64ProgramHeader {
    fn assert_valid(&self) {
        assert_eq!((self.p_vaddr - self.p_offset) % PAGE_SIZE, 0);
        assert_eq!((self.p_paddr - self.p_offset) % PAGE_SIZE, 0);
    }
}

#[repr(C)]
struct Elf64SectionHeader {
    sh_name: u32,
    sh_type: u32,
    sh_flags: u64,
    sh_addr: u64,
    sh_offset: u64,
    sh_size: u64,
    sh_link: u32,
    sh_info: u32,
    sh_addralign: u64,
    sh_entsize: u64,
}