use std::slice;

use crate::{Elf64Sym, Impossible, LinkerOutput};

use cascade::cascade;
use gimli::{
    write::{Address, AttributeValue, DwarfUnit, EndianVec, Range, RangeList, Sections},
    DW_AT_ranges, Encoding, Format, LittleEndian,
};

const PAGE_SIZE: u64 = 0x1000;
const ENTRY: u64 = 0x400000;

const SHF_WRITE: u64 = 0x1;
const SHF_ALLOC: u64 = 0x2;
const SHF_EXECINSTR: u64 = 0x4;

struct Aligned {
    required_padding: u64,
    padding: Vec<u8>,
}

fn align(number: u64, alignment: u64) -> Aligned {
    if alignment == 0 {
        return Aligned {
            required_padding: 0,
            padding: Vec::new(),
        };
    }
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

#[derive(Default)]
struct SH {
    sh_type: u32,
    sh_flags: u64,
    sh_link: u32,
    sh_info: u32,
    sh_entsize: u64,
    sh_addralign: u64,
    content: Vec<u8>,
    content_align: u64,
}

impl SH {
    fn new() -> Self {
        Self {
            sh_type: 1,
            sh_addralign: 1,
            content_align: 0,
            ..Default::default()
        }
    }

    fn table<T>(mut self) -> Self {
        self.sh_entsize = size_of::<T>() as u64;
        self
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
    fn new() -> Self {
        Self {
            names: vec![""],
            ..Default::default()
        }
    }
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
            .flat_map(|name| cascade! { name.as_bytes().to_vec(); ..push(0); })
            .collect();

        let len = self.names.len();

        let shstrtab = SH {
            sh_type: 3,
            content: names,
            sh_addralign: 1,
            ..SH::new()
        };

        self.section(".shstrtab", shstrtab);

        len
    }

    fn generate(self) -> Vec<u8> {
        let mut res = Vec::new();
        let mut file_content = Vec::new();

        let e_shstrndx = self.names.len() as u16 - 1;

        let mut header_size = (size_of::<Elf64Header>()
            + self.loaders.len() * size_of::<Elf64ProgramHeader>()
            + (self.sections.len() + 1) * size_of::<Elf64SectionHeader>())
            as u64;

        let mut header_padding = align(header_size, self.sections[0].sh_addralign);
        header_size += header_padding.required_padding;

        let elf_header_size = size_of::<Elf64Header>() as u64;
        let elf_program_header_size = size_of::<Elf64ProgramHeader>() as u64;

        let elf_header = Elf64Header {
            e_ident: [0x7F, b'E', b'L', b'F', 2, 1, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0],
            e_type: 2,
            e_machine: 0x3E,
            e_version: 1,
            e_entry: ENTRY,
            e_phoff: elf_header_size,
            e_shoff: elf_header_size + self.loaders.len() as u64 * elf_program_header_size,
            e_flags: 0,
            e_ehsize: elf_header_size as u16,
            e_phentsize: elf_program_header_size as u16,
            e_phnum: self.loaders.len() as u16,
            e_shentsize: size_of::<Elf64SectionHeader>() as u16,
            e_shnum: self.sections.len() as u16 + 1,
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
            let content_len = align_vector(&mut content, PAGE_SIZE);

            let addr = ENTRY + n;
            let program_header = Elf64ProgramHeader {
                p_type,
                p_flags,
                p_offset: header_size + n,
                p_vaddr: addr,
                p_paddr: addr,
                p_filesz: content_len,
                p_memsz: content_len,
                p_align: PAGE_SIZE,
            };
            n += content_len;

            extend(&mut res, program_header);
        }

        let name_lengths: Vec<u32> = self
            .names
            .iter()
            .map(|name| name.len() as u32 + 1)
            .collect();

        extend(&mut res, Elf64SectionHeader::default());
        let mut index = 1;
        for SH {
            sh_type,
            sh_flags,
            sh_entsize,
            sh_link,
            sh_info,
            sh_addralign,
            mut content,
            content_align,
        } in self.sections
        {
            align_vector(&mut file_content, sh_addralign);

            let sh_name = name_lengths[..index].iter().sum();
            index += 1;
            let sh_offset = header_size + file_content.len() as u64;
            let mut sh_addr = 0;

            if sh_flags & SHF_ALLOC != 0 {
                sh_addr = ENTRY + file_content.len() as u64;
            }

            let sh_size = content.len() as u64;
            align_vector(&mut content, content_align);

            file_content.append(&mut content);

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
        symtab_sh_info,
    }: LinkerOutput,
    data: Vec<u8>,
) -> Vec<u8> {
    let mut elf_generator = ElfGenerator::new();

    elf_generator.load(1, 1 | 4, text.clone());
    elf_generator.load(1, 2 | 4, data.clone());

    // .text
    {
        let text_section = SH {
            sh_flags: SHF_ALLOC | SHF_EXECINSTR,
            sh_addralign: PAGE_SIZE,
            content_align: PAGE_SIZE,
            content: text,
            ..SH::new()
        };
        elf_generator.section(".text", text_section);
    }

    // .data
    {
        let data_section = SH {
            sh_flags: SHF_WRITE | SHF_ALLOC,
            sh_addralign: 4,
            content: data,
            ..SH::new()
        };
        elf_generator.section(".data", data_section);
    }

    // .debug_info
    {
        let encoding = Encoding {
            address_size: 8,
            format: Format::Dwarf64,
            version: 5,
        };

        let mut dwarf = DwarfUnit::new(encoding);
        let range_list = RangeList(vec![Range::StartLength {
            // FIXME:
            begin: Address::Constant(0x100),
            length: 42,
        }]);

        let range_list_id = dwarf.unit.ranges.add(range_list);
        dwarf
            .unit
            .get_mut(dwarf.unit.root())
            .set(DW_AT_ranges, AttributeValue::RangeListRef(range_list_id));

        let mut sections = Sections::new(EndianVec::new(LittleEndian));
        dwarf.write(&mut sections).impossible().unwrap();

        let mut content = Vec::new();
        sections
            .for_each(|_, data| {
                content.append(&mut data.clone().into_vec());
                Result::<(), ()>::Ok(())
            })
            .unwrap();

        let debug_info_section = SH {
            content,
            sh_type: 1,
            ..Default::default()
        };

        elf_generator.section(".debug_info", debug_info_section);
    }

    // .symtab
    {
        let mut symtab_content = Vec::new();
        for symbol in symtab {
            extend(&mut symtab_content, symbol);
        }

        let symtab = SH {
            sh_type: 2,
            sh_link: elf_generator.sections.len() as u32 + 2,
            sh_info: symtab_sh_info,
            sh_addralign: 8,
            content: symtab_content,
            ..SH::new()
        }
        .table::<Elf64Sym>();

        elf_generator.section(".symtab", symtab);
    }

    // .strtab
    {
        let strtab_content = strtab
            .into_iter()
            .flat_map(|string| string.as_bytes().to_vec())
            .collect();

        let strtab = SH {
            sh_type: 3,
            content: strtab_content,
            ..SH::new()
        };
        elf_generator.section(".strtab", strtab);
    }

    elf_generator.generate_shstrtab();

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

#[derive(Default)]
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
