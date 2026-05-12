// Need to replace in template:
//  {{ num_points }}
//  {{ node_workgroup_size_x }}
//  {{ node_workgroup_size_y }}
//  {{ node_workgroup_size_z }}
//
//  {{ num_cells }}
//  {{ cell_workgroup_size_x }}
//  {{ cell_workgroup_size_y }}
//  {{ cell_workgroup_size_z }}
//
//  {{ max_points_per_cell }}
//  {{ search_radius }}


// Point-specific data (length = num_points)
@group(0) @binding(0)
var<storage, read> positions: array<vec3<f32>>;

@group(0) @binding(1)
var<storage, read> point_data: array<f32>;

@group(0) @binding(2)
var<storage, read_write> neighbor_sum: array<f32>;


// Cell-specific data (length = num_cells)
@group(1) @binding(0)
var<storage, read_write> cell_counts: array<atomic<u32>>;

@group(1) @binding(1)
var<storage, read_write> cell_point_indices: array<u32>;

fn compute_cell_index(pos: vec3<f32>) -> u32 {
    return hash_cell_position(cell_position(pos));
}

fn cell_position(pos: vec3<f32>) -> vec3<i32> {
    return vec3<i32>(floor(pos / {{ search_radius }}));
}

fn hash_cell_position(cell_pos: vec3<i32>) -> u32 {
    // Locally perfect hash from: https://haug.codes/blog/locally-perfect-hashing/
    let kappa = (cell_pos.x % 3) + 3*(cell_pos.y % 3) + 9*(cell_pos.z % 3);
    let beta = (u32(cell_pos.x) * 73856093 ^ u32(cell_pos.y) * 19349663 ^ u32(cell_pos.z) * 83492791) % {{ num_cells }};
    return (beta / 27) * 27 + u32(kappa);

    // // Hash: Teschner, Matthias, Bruno Heidelberger, Matthias Müller, Danat Pomerantes, and Markus H. Gross. "Optimized spatial hashing for collision detection of deformable objects." In Vmv, vol. 3, pp. 47-54. 2003.
    // // Note: this is not locally perfect
    // return (u32(cell_pos.x) * 73856093 ^ u32(cell_pos.y) * 19349663 ^ u32(cell_pos.z) * 83492791) % {{ num_cells }};
}

// Populates cells
@compute @workgroup_size({{ point_workgroup_size_x }}, {{ point_workgroup_size_y }}, {{ point_workgroup_size_z }})
fn build_cells(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x + global_id.y * {{ point_workgroup_size_x }} + global_id.z * {{ point_workgroup_size_x }} * {{ point_workgroup_size_y }};
    if (i >= {{ num_points }}) { return; }

    let pos = positions[i];
    let cell_index = compute_cell_index(pos);
    let count = atomicAdd(&cell_counts[cell_index], 1);

    // Do not write if cell is full, counts can be higher than max_points_per_cell!!!
    if (count >= {{ max_points_per_cell }}) { return; }
    cell_point_indices[cell_index * {{ max_points_per_cell }} + count] = i;
}

// Simply adds the "point_data" of neighboring points
@compute @workgroup_size({{ point_workgroup_size_x }}, {{ point_workgroup_size_y }}, {{ point_workgroup_size_z }})
fn compute_neighbor_sums(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x + global_id.y * {{ point_workgroup_size_x }} + global_id.z * {{ point_workgroup_size_x }} * {{ point_workgroup_size_y }};
    if (i >= {{ num_points }}) { return; }

    let pos = positions[i];
    let cell_position = cell_position(pos);

    var total = 0.0;

    for (var cell_x: i32 = -1; cell_x <= 1; cell_x++) {
        for (var cell_y: i32 = -1; cell_y <= 1; cell_y++) {
            for (var cell_z: i32 = -1; cell_z <= 1; cell_z++) {
                let neighbor_cell_index = hash_cell_position(cell_position + vec3<i32>(cell_x, cell_y, cell_z));
                let count = min(cell_counts[neighbor_cell_index], {{ max_points_per_cell }});
                for (var j: u32 = 0; j < count; j++) {
                    let j = cell_point_indices[neighbor_cell_index * {{ max_points_per_cell }} + j];
                    // Do something with neighbor_index
                    if length(positions[j] - pos) < {{ search_radius }} {
                        // total += point_data[j];
                        total += 1.0;
                    }
                }
            }
        }
    }
    neighbor_sum[i] = total;
}