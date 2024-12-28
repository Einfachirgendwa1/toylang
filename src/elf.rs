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

#[derive(Default)]
struct ElfGenerator {
    sections: Vec<(&'static str, u32, u64, Option<Vec<u8>>)>,
    loaders: Vec<(u32, u32, Vec<u8>)>,
}

impl ElfGenerator {
    fn section(&mut self, name: &'static str, sh_type: u32, sh_flags: u64) {
        self.sections.push((name, sh_type, sh_flags, None));
    }

    fn section_with_content(
        &mut self,
        name: &'static str,
        sh_type: u32,
        sh_flags: u64,
        content: Vec<u8>,
    ) {
        self.sections.push((name, sh_type, sh_flags, Some(content)));
    }

    fn load(&mut self, p_type: u32, p_flags: u32, content: Vec<u8>) {
        self.loaders.push((p_type, p_flags, content));
    }

    fn generate(self) -> Vec<u8> {
        let mut res = Vec::new();
        let mut file_content = Vec::new();

        let names: Vec<u32> = self
            .sections
            .iter()
            .map(|(a, _, _, _)| a.len() as u32)
            .collect();

        let mut header_size = (size_of::<Elf64Header>()
            + self.loaders.len() * size_of::<Elf64ProgramHeader>()
            + self.sections.len() * size_of::<Elf64SectionHeader>())
            as u64;

        let mut header_padding = align(header_size, ALIGNMENT);
        header_size += header_padding.required_padding;

        let elf_header_size = size_of::<Elf64Header>() as u64;
        let elf_program_header_size = size_of::<Elf64ProgramHeader>() as u64;

        let elf_header = Elf64Header {
            e_ident: [0x7F, b'E', b'L', b'F', 2, 1, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0],
            e_type: 2,
            e_machine: 0x3E,
            e_version: 1,
            e_entry: 0x400000,
            e_phoff: elf_header_size,
            e_shoff: elf_header_size + self.loaders.len() as u64 * elf_program_header_size,
            e_flags: 0,
            e_ehsize: elf_header_size as u16,
            e_phentsize: elf_program_header_size as u16,
            e_phnum: self.loaders.len() as u16,
            e_shentsize: size_of::<Elf64SectionHeader>() as u16,
            e_shnum: self.sections.len() as u16,
            e_shstrndx: self
                .sections
                .iter()
                .enumerate()
                .find_map(|(index, (x, _, _, _))| (*x == ".shstrtab\0").then(|| index as u16))
                .unwrap(),
        };

        extend(&mut res, elf_header);

        let mut n = 0;
        for (p_type, p_flags, mut content) in self.loaders {
            let content_len = align_vector(&mut content, ALIGNMENT);

            let addr = n + 0x400000;
            let program_header = Elf64ProgramHeader {
                p_type,
                p_flags,
                p_offset: header_size + n as u64,
                p_vaddr: addr,
                p_paddr: addr,
                p_filesz: content_len,
                p_memsz: content_len,
                p_align: ALIGNMENT,
            };
            n += content_len;

            extend(&mut res, program_header);
        }

        let mut n = 0;
        for (_, sh_type, sh_flags, content) in self.sections {
            let sh_name = names[..n].iter().sum();
            n += 1;

            let mut sh_addr = 0;
            let mut sh_offset = 0;
            let mut sh_size = 0;

            if let Some(mut content) = content {
                sh_addr = 0x400000 + file_content.len() as u64;
                sh_offset = header_size + file_content.len() as u64;
                sh_size = content.len() as u64;

                file_content.append(&mut content);
            }

            let section_header = Elf64SectionHeader {
                sh_name,
                sh_type,
                sh_flags,
                sh_addr,
                sh_offset,
                sh_link: 0,
                sh_size,
                sh_info: 0,
                sh_entsize: 0,
                sh_addralign: ALIGNMENT,
            };

            extend(&mut res, section_header);
        }

        res.append(&mut header_padding.padding);
        res.append(&mut file_content);

        res
    }
}

pub fn generate_elf(text: Vec<u8>, data: Vec<u8>) -> Vec<u8> {
    let sections = ["\0", ".text\0", ".data\0", ".shstrtab\0"];

    let mut elf_generator = ElfGenerator::default();

    elf_generator.load(1, 1 | 4, text.clone());
    elf_generator.load(1, 2 | 4, data.clone());

    elf_generator.section("\0", 0, 0);
    elf_generator.section_with_content(".text\0", 1, 2 | 4, text);
    elf_generator.section_with_content(".data\0", 1, 1 | 2, data);
    elf_generator.section_with_content(".shstrtab\0", 3, 0, as_vec_u8(&sections));

    elf_generator.generate()
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
