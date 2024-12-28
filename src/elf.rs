use crate::{Elf64Sym, LinkerOutput};
use std::slice;

const LOAD_ALIGNMENT: u64 = 0x1000;
const BULLSHIT_ALIGNMENT: u64 = 4;

struct Aligned {
    required_padding: u64,
    padding: Vec<u8>,
}

fn align(number: u64, alignment: u64) -> Aligned {
    if number % alignment == 0 {
        return Aligned {
            required_padding: 0,
            padding: Vec::new(),
        };
    }
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

fn extend<A: Clone, B>(vec: &mut Vec<A>, b: B) {
    unsafe {
        vec.extend_from_slice(slice::from_raw_parts(
            &b as *const B as *const A,
            size_of::<B>(),
        ))
    }
}

struct SH {
    sh_type: u32,
    sh_flags: u64,
    sh_link: u32,
    sh_info: u32,
    sh_entsize: u64,
    content: Option<Vec<u8>>,
}

impl SH {
    fn no_content(sh_type: u32, sh_flags: u64) -> SH {
        SH {
            sh_type,
            sh_flags,
            sh_link: 0,
            content: None,
            sh_info: 0,
            sh_entsize: 0,
        }
    }

    fn no_table(sh_type: u32, sh_flags: u64, content: Vec<u8>) -> SH {
        SH {
            sh_type,
            sh_flags,
            sh_link: 0,
            sh_info: 0,
            content: Some(content),
            sh_entsize: 0,
        }
    }

    fn table<T>(sh_type: u32, sh_flags: u64, content: Vec<u8>) -> SH {
        SH {
            sh_type,
            sh_flags,
            sh_link: 0,
            sh_info: 0,
            content: Some(content),
            sh_entsize: size_of::<T>() as u64,
        }
    }
}

struct PH {
    p_type: u32,
    p_flags: u32,
    content: Vec<u8>,
}

#[derive(Default)]
struct ElfGenerator {
    names: Vec<&'static str>,
    sections: Vec<SH>,
    loaders: Vec<PH>,
}

impl ElfGenerator {
    fn section(&mut self, name: &'static str, sh: SH) {
        self.sections.push(sh);
        self.names.push(name);
    }

    fn load(&mut self, p_type: u32, p_flags: u32, content: Vec<u8>) {
        self.loaders.push(PH {
            p_type,
            p_flags,
            content,
        });
    }

    fn generate_shstrtab(&mut self) -> usize {
        let mut names = self.names.clone();

        names.push(".shstrtab");

        let names: Vec<u8> = names
            .into_iter()
            .flat_map(|name| name.as_bytes().to_vec())
            .collect();

        let len = self.names.len();
        self.section(".shstrtab", SH::no_table(3, 0, names));

        len
    }

    fn find_section_by_name(&self, section_name: &'static str) -> Option<usize> {
        self.names
            .iter()
            .enumerate()
            .find_map(|(index, name)| (*name == section_name).then(|| index))
    }

    fn generate(mut self) -> Vec<u8> {
        let mut res = Vec::new();
        let mut file_content = Vec::new();

        let e_shstrndx = self.generate_shstrtab() as u16;

        let mut header_size = (size_of::<Elf64Header>()
            + self.loaders.len() * size_of::<Elf64ProgramHeader>()
            + self.sections.len() * size_of::<Elf64SectionHeader>())
            as u64;

        let mut header_padding = align(header_size, LOAD_ALIGNMENT);
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
            e_shstrndx,
        };

        extend(&mut res, elf_header);

        let mut n = 0;
        for PH {
            p_type,
            p_flags,
            mut content,
        } in self.loaders
        {
            let content_len = align_vector(&mut content, LOAD_ALIGNMENT);

            let addr = n + 0x400000;
            let program_header = Elf64ProgramHeader {
                p_type,
                p_flags,
                p_offset: header_size + n as u64,
                p_vaddr: addr,
                p_paddr: addr,
                p_filesz: content_len,
                p_memsz: content_len,
                p_align: LOAD_ALIGNMENT,
            };
            n += content_len;

            extend(&mut res, program_header);
        }

        let name_bytes: Vec<u32> = self.names.iter().map(|name| name.len() as u32).collect();

        let mut n = 0;
        for SH {
            sh_type,
            sh_flags,
            sh_entsize,
            content,
            sh_link,
            sh_info,
        } in self.sections
        {
            let sh_name = name_bytes[..n].iter().sum();
            n += 1;

            let mut sh_addr = 0;
            let sh_offset = header_size + file_content.len() as u64;
            let mut sh_size = 0;

            let mut sh_addralign = BULLSHIT_ALIGNMENT;

            if let Some(mut content) = content {
                if sh_flags & 2 == 2 {
                    sh_addralign = LOAD_ALIGNMENT;
                    sh_addr = 0x400000 + file_content.len() as u64;
                }
                align_vector(&mut content, sh_addralign);

                sh_size = content.len() as u64;

                file_content.append(&mut content);
            }

            let section_header = Elf64SectionHeader {
                sh_name,
                sh_type,
                sh_flags,
                sh_addr,
                sh_offset,
                sh_link,
                sh_size,
                sh_info,
                sh_entsize,
                sh_addralign,
            };

            extend(&mut res, section_header);
        }

        res.append(&mut header_padding.padding);
        res.append(&mut file_content);

        res
    }
}

pub fn generate_elf(
    LinkerOutput {
        code: text,
        symtab,
        strtab,
    }: LinkerOutput,
    data: Vec<u8>,
) -> Vec<u8> {
    let mut elf_generator = ElfGenerator::default();

    elf_generator.load(1, 1 | 4, text.clone());
    elf_generator.load(1, 2 | 4, data.clone());

    elf_generator.section("\0", SH::no_content(0, 0));
    elf_generator.section(".text\0", SH::no_table(1, 2 | 4, text));
    elf_generator.section(".data\0", SH::no_table(1, 1 | 2, data));

    let strtab_content = strtab
        .into_iter()
        .flat_map(|mut string: String| {
            println!("{string}");
            string.push('\0');
            string.as_bytes().to_vec()
        })
        .collect();

    elf_generator.section(".strtab\0", SH::no_table(3, 0, strtab_content));

    let mut symtab_content = Vec::new();
    for symbol in symtab {
        extend(&mut symtab_content, symbol);
    }

    let sh_info = symtab_content.len() as u32;
    let mut symtab = SH::table::<Elf64Sym>(2, 0, symtab_content);
    symtab.sh_link = elf_generator.find_section_by_name(".strtab\0").unwrap() as u32;
    symtab.sh_info = sh_info;

    elf_generator.section(".symtab\0", symtab);

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
