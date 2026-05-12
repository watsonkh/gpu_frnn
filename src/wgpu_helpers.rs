use wgpu::util::DeviceExt;

pub(crate) fn create_buffer(
    gpu: &wgpu::Device,
    size: u64,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    gpu.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage,
        mapped_at_creation: false,
    })
}

pub(crate) fn create_buffer_init<T: bytemuck::Pod>(
    gpu: &wgpu::Device,
    contents: &[T],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    gpu.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(contents),
        usage,
    })
}

pub(crate) fn bind_group_layout_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

pub(crate) fn bind_group_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}
