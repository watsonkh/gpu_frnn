use pollster::block_on;
use rand::RngExt;

mod wgpu_helpers;
use wgpu_helpers::{bind_group_entry, bind_group_layout_entry, create_buffer_init};

use wgpu::BufferUsages;

fn main() {
    let (gpu, queue) = gpu_boilerplate();

    let scale: u32 = 100;
    // a density of 1 point per unit^3, for a scale of 40, we get 40^3 = 64000 points
    let num_points = scale.pow(3) as usize;
    let search_radius = 1.0;

    let max_points_per_cell = 10;
    // let num_cells = num_points.next_multiple_of(27); // For perfect local hashing, must be a multiple of 27 for 3D, or 9 for 2D, etc.
    let num_cells = (num_points * 2).next_power_of_two(); // For perfect local hashing, must be a multiple of 27 for 3D, or 9 for 2D, etc.

    let cell_workgroup_size_x = 16;
    let cell_workgroup_size_y = 1;
    let cell_workgroup_size_z = 1;

    let point_workgroup_size_x = 16;
    let point_workgroup_size_y = 1;
    let point_workgroup_size_z = 1;

    let mut positions_cpu = Vec::with_capacity(num_points);
    for x in 0..scale {
        for y in 0..scale {
            for z in 0..scale {
                positions_cpu.push([x as f32, y as f32, z as f32, 0.0]);
            }
        }
    }
    // println!("{:?}", positions_cpu);
    // let positions_cpu = (0..num_points)
    //     .map(|_| rand::rng().random::<[f32; 4]>())
    //     .map(|pos| {
    //         let mut pos = pos;
    //         pos[0] *= scale as f32;
    //         pos[1] *= scale as f32;
    //         pos[2] *= scale as f32;
    //         pos[3] = 0.0;
    //         pos
    //     })
    //     .collect::<Vec<[f32; 4]>>();

    // Buffers for group 0
    let positions = create_buffer_init(
        &gpu,
        &positions_cpu,
        BufferUsages::STORAGE | BufferUsages::COPY_DST,
    );
    let neighbor_counts = create_buffer_init(
        &gpu,
        &vec![0.0f32; num_points as usize],
        BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
    );

    // Bind group for group 0
    let point_bg_layout = gpu.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("point_bg_layout"),
        entries: &[
            bind_group_layout_entry(0, true),
            bind_group_layout_entry(2, false),
        ],
    });
    let point_bind_group = gpu.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("point_bind_group"),
        layout: &point_bg_layout,
        entries: &[
            bind_group_entry(0, &positions),
            bind_group_entry(2, &neighbor_counts),
        ],
    });

    // Buffers for group 1
    let cell_counts = create_buffer_init(
        &gpu,
        &vec![0u32; num_cells],
        BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
    );
    let cell_indices = create_buffer_init(
        &gpu,
        &vec![0u32; max_points_per_cell * num_cells],
        BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
    );

    // Bind group for group 1
    let cell_bg_layout = gpu.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("cell_bg_layout"),
        entries: &[
            bind_group_layout_entry(0, false),
            bind_group_layout_entry(1, false),
        ],
    });
    let cell_bind_group = gpu.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cell_bind_group"),
        layout: &cell_bg_layout,
        entries: &[
            bind_group_entry(0, &cell_counts),
            bind_group_entry(1, &cell_indices),
        ],
    });

    // Parse, specialize, and compile shader
    let raw_shader = include_str!("shaders/frnn.wgsl")
        .to_string()
        .replace("{{ num_points }}", &(num_points.to_string() + "u"))
        .replace("{{ num_cells }}", &(num_cells.to_string() + "u"))
        .replace("{{ search_radius }}", &search_radius.to_string())
        .replace(
            "{{ max_points_per_cell }}",
            &(max_points_per_cell.to_string() + "u"),
        )
        .replace(
            "{{ point_workgroup_size_x }}",
            &point_workgroup_size_x.to_string(),
        )
        .replace(
            "{{ point_workgroup_size_y }}",
            &point_workgroup_size_y.to_string(),
        )
        .replace(
            "{{ point_workgroup_size_z }}",
            &point_workgroup_size_z.to_string(),
        )
        .replace(
            "{{ cell_workgroup_size_x }}",
            &cell_workgroup_size_x.to_string(),
        )
        .replace(
            "{{ cell_workgroup_size_y }}",
            &cell_workgroup_size_y.to_string(),
        )
        .replace(
            "{{ cell_workgroup_size_z }}",
            &cell_workgroup_size_z.to_string(),
        );
    let shader = gpu.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("shader"),
        source: wgpu::ShaderSource::Wgsl(raw_shader.into()),
    });

    // Compute pipelines
    let pipeline_layout = gpu.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pipeline_layout"),
        bind_group_layouts: &[Some(&point_bg_layout), Some(&cell_bg_layout)],
        immediate_size: 0,
    });
    let build_cells_pipeline = gpu.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("build_cells_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("build_cells"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let compute_neighbor_sums_pipeline =
        gpu.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("compute_neighbor_sums_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("compute_neighbor_sums"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

    // Run pipelines
    let mut command_encoder = gpu.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("command_encoder"),
    });
    // Clear counts buffer
    command_encoder.clear_buffer(&cell_counts, 0, None);
    {
        let mut pass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compute Pass"),
            timestamp_writes: None,
        });

        pass.set_bind_group(0, &point_bind_group, &[]);
        pass.set_bind_group(1, &cell_bind_group, &[]);

        // Create cells
        pass.set_pipeline(&build_cells_pipeline);
        pass.dispatch_workgroups(
            num_points.div_ceil(point_workgroup_size_x) as u32,
            point_workgroup_size_y,
            point_workgroup_size_z,
        );

        // Compute neighbor sums
        pass.set_pipeline(&compute_neighbor_sums_pipeline);
        pass.dispatch_workgroups(
            num_points.div_ceil(point_workgroup_size_x) as u32,
            point_workgroup_size_y,
            point_workgroup_size_z,
        );
    }
    queue.submit([command_encoder.finish()]);
    let now = std::time::Instant::now();
    let _ = gpu.poll(wgpu::PollType::wait_indefinitely());
    println!("{:?}", std::time::Instant::now() - now);

    // Read results
    let staging_buffer = create_buffer_init(
        &gpu,
        &vec![0.0f32; num_points as usize],
        BufferUsages::MAP_READ | BufferUsages::COPY_DST,
    );
    let mut command_encoder = gpu.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Second Command Encoder"),
    });
    command_encoder.copy_buffer_to_buffer(
        &neighbor_counts,
        0,
        &staging_buffer,
        0,
        (num_points * std::mem::size_of::<f32>()) as u64,
    );
    queue.submit([command_encoder.finish()]);
    let _ = gpu.poll(wgpu::PollType::wait_indefinitely());
    let buffer_slice = staging_buffer.slice(..);
    buffer_slice.map_async(wgpu::MapMode::Read, |_| ());
    let _ = gpu.poll(wgpu::PollType::wait_indefinitely());
    let data = buffer_slice.get_mapped_range();
    let neighbor_sums: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging_buffer.unmap();

    // println!("{:?}", neighbor_sums);

    // let tree = rstar::RTree::bulk_load(
    //     positions_cpu
    //         .iter()
    //         .enumerate()
    //         .map(|(i, pos)| rstar::primitives::GeomWithData::new([pos[0], pos[1], pos[2]], i))
    //         .collect::<Vec<_>>(),
    // );

    // let actual_neighbor_count = (0..num_points)
    //     .map(|i| {
    //         tree.locate_within_distance(
    //             [
    //                 positions_cpu[i][0],
    //                 positions_cpu[i][1],
    //                 positions_cpu[i][2],
    //             ],
    //             search_radius * search_radius,
    //         )
    //         .collect::<Vec<_>>()
    //         .iter()
    //         .count() as u32
    //     })
    //     .collect::<Vec<u32>>();

    // // for (i, sum) in neighbor_sums.iter().enumerate() {
    // //     println!("{}, {}", *sum as u32, actual_neighbor_count[i]);
    // // }
    // for (i, sum) in neighbor_sums.iter().enumerate() {
    //     assert_eq!(*sum as u32, actual_neighbor_count[i]);
    // }
    // println!("Done!");

    // Read cell counts buffer (should not go over max_points_per_cell)
    let staging_buffer = create_buffer_init(
        &gpu,
        &vec![0u32; num_cells],
        BufferUsages::MAP_READ | BufferUsages::COPY_DST,
    );
    let mut command_encoder = gpu.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Second Command Encoder"),
    });
    command_encoder.copy_buffer_to_buffer(
        &cell_counts,
        0,
        &staging_buffer,
        0,
        (num_cells * std::mem::size_of::<u32>()) as u64,
    );
    queue.submit([command_encoder.finish()]);
    let _ = gpu.poll(wgpu::PollType::wait_indefinitely());
    let buffer_slice = staging_buffer.slice(..);
    buffer_slice.map_async(wgpu::MapMode::Read, |_| ());
    let _ = gpu.poll(wgpu::PollType::wait_indefinitely());
    let data = buffer_slice.get_mapped_range();
    let cell_counts: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging_buffer.unmap();

    // println!("cell counts: {:?}", cell_counts);
    println!("Number of cells: {}", num_cells);
    println!("Number of points: {}", num_points);
    println!("Max cell count: {}", cell_counts.iter().max().unwrap());
    println!(
        "Sparse cell ratio (empty / total): {}",
        cell_counts.iter().filter(|cell| **cell == 0).count() as f64 / num_cells as f64
    );
    println!("Points cell count: {}", cell_counts.iter().sum::<u32>());
}

fn gpu_boilerplate() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        flags: wgpu::InstanceFlags::empty(),
        memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
        backend_options: wgpu::BackendOptions::default(),
        display: None,
    });

    println!("Available adapters:");
    for name in block_on(instance.enumerate_adapters(wgpu::Backends::all()))
        .iter()
        .map(|adapter| {
            let info = adapter.get_info();
            (
                info.name,
                info.backend,
                info.device_type,
                info.subgroup_max_size,
                info.subgroup_min_size,
            )
        })
        .collect::<Vec<_>>()
    {
        println!("{:?}", name);
    }
    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("Failed to find an appropriate adapter");

    let info = adapter.get_info();
    // println!("Using adapter: {:?}", adapter.get_info());
    println!(
        "\nUsing adapter: {:?}",
        (
            info.name,
            info.backend.to_string(),
            info.device_type,
            info.subgroup_max_size,
            info.subgroup_min_size,
        )
    );

    let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("Device and Queue"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
        experimental_features: wgpu::ExperimentalFeatures::default(),
        memory_hints: wgpu::MemoryHints::default(),
        trace: wgpu::Trace::Off,
    }))
    .expect("Failed to create device");

    (device, queue)
}
