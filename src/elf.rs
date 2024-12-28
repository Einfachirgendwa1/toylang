use std::slice;

const ALIGNMENT: u64 = 0x1000;

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

pub fn extend<A: Clone, B>(vec: &mut Vec<A>, b: B) {
    unsafe {
        vec.extend_from_slice(slice::from_raw_parts(
            &b as *const B as *const A,
            size_of::<B>(),
        ))
    }
}

fn as_vec_u8(slice: &[&str]) -> Vec<u8> {
    slice.iter().flat_map(|x| x.as_bytes().to_vec()).collect()
}

struct ElfBuilder<const N: usize> {
    section_indezes: [u32; N],
    pp_count: usize,
    rp_count: usize,
    loader_n: usize,
    n: usize,
    header_size: u64,
    header_padding: Vec<u8>,
    headers: Vec<u8>,
    content: Vec<u8>,
}

impl<const N: usize> ElfBuilder<N> {
    fn new(sections: [&'static str; N], program_headers: usize) -> Self {
        let mut header_size = (size_of::<Elf64Header>()
            + program_headers * size_of::<Elf64ProgramHeader>()
            + sections.len() * size_of::<Elf64SectionHeader>())
            as u64;
        let header_padding = align(header_size, ALIGNMENT);
        header_size += header_padding.required_padding;

        let mut headers = Vec::new();

        let elf_header_size = size_of::<Elf64Header>() as u64;
        let elf_program_header_size = size_of::<Elf64ProgramHeader>() as u64;

        let elf_header = Elf64Header {
            e_ident: [0x7F, b'E', b'L', b'F', 2, 1, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0],
            e_type: 2,
            e_machine: 0x3E,
            e_version: 1,
            e_entry: 0x400000,
            e_phoff: elf_header_size,
            e_shoff: elf_header_size + 2 * elf_program_header_size,
            e_flags: 0,
            e_ehsize: elf_header_size as u16,
            e_phentsize: elf_program_header_size as u16,
            e_phnum: 2,
            e_shentsize: size_of::<Elf64SectionHeader>() as u16,
            e_shnum: 4,
            e_shstrndx: 3,
        };

        extend(&mut headers, elf_header);

        Self {
            pp_count: program_headers,
            rp_count: 0,
            header_padding: header_padding.padding,
            section_indezes: sections.map(|section| section.len() as u32),
            n: 0,
            loader_n: 0,
            header_size,
            headers,
            content: Vec::new(),
        }
    }

    fn add_p(&mut self, p_type: u32, p_flags: u32, data: &mut Vec<u8>) {
        let data_len = align_vector(data, ALIGNMENT);
        self.rp_count += 1;

        let addr = self.loader_n as u64 + 0x400000;
        let program_header = Elf64ProgramHeader {
            p_type,
            p_flags,
            p_offset: self.header_size + self.loader_n as u64,
            p_vaddr: addr,
            p_paddr: addr,
            p_filesz: data_len,
            p_memsz: data_len,
            p_align: ALIGNMENT,
        };

        self.loader_n += data_len as usize;

        extend(&mut self.headers, program_header);
    }

    fn add_sh(&mut self, sh_flags: u64, sh_type: u32) {
        let sh_name = self.section_indezes[..self.n].iter().sum();
        self.n += 1;

        let section_header = Elf64SectionHeader {
            sh_name,
            sh_type,
            sh_flags,
            sh_addr: 0,
            sh_offset: 0,
            sh_link: 0,
            sh_size: 0,
            sh_info: 0,
            sh_entsize: 0,
            sh_addralign: ALIGNMENT,
        };

        extend(&mut self.headers, section_header);
    }

    fn add_sh_with_content(&mut self, sh_type: u32, sh_flags: u64, mut content: Vec<u8>) {
        let sh_name = self.section_indezes[..self.n].iter().sum();
        self.n += 1;

        let section_header = Elf64SectionHeader {
            sh_name,
            sh_type,
            sh_flags,
            sh_addr: 0x400000 + self.content.len() as u64,
            sh_offset: self.header_size + self.content.len() as u64,
            sh_link: 0,
            sh_size: content.len() as u64,
            sh_info: 0,
            sh_entsize: 0,
            sh_addralign: ALIGNMENT,
        };

        extend(&mut self.headers, section_header);
        self.content.append(&mut content);
    }

    fn finish(mut self) -> Vec<u8> {
        debug_assert_eq!(self.rp_count, self.pp_count);
        debug_assert_eq!(self.n, N);

        let mut res = self.headers;
        res.append(&mut self.header_padding);
        res.append(&mut self.content);
        res
    }
}

pub fn generate_elf(mut text: Vec<u8>, mut data: Vec<u8>) -> Vec<u8> {
    let sections = ["\0", ".text\0", ".data\0", ".shstrtab\0"];

    let mut elf_builder = ElfBuilder::new(sections, 2);

    elf_builder.add_p(1, 1 | 4, &mut text);
    elf_builder.add_p(1, 2 | 4, &mut data);

    elf_builder.add_sh(0, 0);
    elf_builder.add_sh_with_content(1, 2 | 4, text);
    elf_builder.add_sh_with_content(1, 1 | 2, data);
    elf_builder.add_sh_with_content(3, 0, as_vec_u8(&sections));

    elf_builder.finish()
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
