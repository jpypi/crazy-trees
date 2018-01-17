use std::cmp::Eq;
use std::hash::Hash;
use std::collections::HashMap;

type Num = f32;


fn count_dist<T: Hash + Eq + Clone>(data: &Vec<T>) -> Vec<Num> {
    let mut hm = HashMap::new();

    // Count all the instances of something
    for val in data {
        *hm.entry(val).or_insert(0.0) += 1.0;
    }

    // Force a proper sum to 1 distribution
    let ttl = data.len() as Num;
    for val in hm.values_mut() {
        *val /= ttl;
    }

    // Move the values to a vec
    hm.into_iter().map(|(_, v)| v).collect()
}

fn gini(dist: &Vec<Num>) -> Num {
    dist.iter().fold(1.0, |acc, &x| acc - x*x)
}

fn entropy(dist: &Vec<Num>) -> Num {
    -dist.iter().fold(0.0, |acc, &x| acc + x * x.log2())
}


fn set_entropy(data: &[Vec<Num>], label_i: usize) -> Num {
    let labels = data.iter().map(|x| x[label_i] as i32).collect();
    entropy(&count_dist(&labels))
}


fn feature_split(data: &mut Vec<Vec<Num>>, feature_i: usize, label_i: usize) -> (Num, Num) {
    // Sort the data by the feature
    data.sort_by(|a, b| a[feature_i].partial_cmp(&b[feature_i]).unwrap());

    // Calculate the current entropy before any split
    let parent_entropy = set_entropy(data, label_i);

    let (mut best_split_point, mut best_info_gain) = (0.0, 0.0);

    for i in 0..(data.len() - 1) {
        let split_point = (data[i][feature_i] + data[i+1][feature_i]) / 2.0;

        let left  = &data[..i+1];
        let right = &data[i+1..];

        let left_entropy = set_entropy(left, label_i);
        let right_entropy = set_entropy(right, label_i);

        let ttl_entropy = ((i+1) as f32 * left_entropy +
                           ((data.len()-(i+1)) as f32) * right_entropy) /
                          (data.len() as f32);

        let info_gain = parent_entropy - ttl_entropy;

        if info_gain > best_info_gain {
            best_info_gain = info_gain;
            best_split_point = split_point;
        }
    }

    (best_info_gain, best_split_point)
}


fn calc_split(data: &mut Vec<Vec<Num>>, label_i: usize) -> (Num, Num, usize) {
    let (mut best_ig, mut best_sp, mut best_i) = (0.0, 0.0, 0);

    // Loop through features to find the best feature to use
    for i in 0..data[0].len() {
        if i != label_i {
            println!("Calculating feature split on feature {}", i);
            let (ig, sp) = feature_split(data, i, label_i);
            println!("Information gain: {} | Split point: {}", ig, sp);
            if ig > best_ig {
                best_ig = ig;
                best_sp = sp;
                best_i = i;
            }
        }
    }

    (best_ig, best_sp, best_i)
}


/*
fn fit_tree(data: &mut Vec<Vec<Num>>, label_i: usize) {

    calc_split(data, label_i)
}
*/


fn main() {
    let mut sample_data = vec![
        vec![2.771244718, 1.784783929, 0.0],
        vec![1.728571309, 1.169761413, 0.0],
        vec![3.678319846, 2.81281357,  0.0],
        vec![3.961043357, 2.61995032,  0.0],
        vec![2.999208922, 2.209014212, 0.0],
        vec![7.497545867, 3.162953546, 1.0],
        vec![9.00220326 , 3.339047188, 1.0],
        vec![7.444542326, 0.476683375, 1.0],
        vec![10.12493903, 3.234550982, 1.0],
        vec![6.642287351, 3.319983761, 1.0],
    ];

    let mut iris_data = vec![
        vec![5.1,3.5,1.4,0.2,1.0],
        vec![4.9,3.0,1.4,0.2,1.0],
        vec![4.7,3.2,1.3,0.2,1.0],
        vec![4.6,3.1,1.5,0.2,1.0],
        vec![5.0,3.6,1.4,0.2,1.0],
        vec![5.4,3.9,1.7,0.4,1.0],
        vec![4.6,3.4,1.4,0.3,1.0],
        vec![5.0,3.4,1.5,0.2,1.0],
        vec![4.4,2.9,1.4,0.2,1.0],
        vec![4.9,3.1,1.5,0.1,1.0],
        vec![5.4,3.7,1.5,0.2,1.0],
        vec![4.8,3.4,1.6,0.2,1.0],
        vec![4.8,3.0,1.4,0.1,1.0],
        vec![4.3,3.0,1.1,0.1,1.0],
        vec![5.8,4.0,1.2,0.2,1.0],
        vec![5.7,4.4,1.5,0.4,1.0],
        vec![5.4,3.9,1.3,0.4,1.0],
        vec![5.1,3.5,1.4,0.3,1.0],
        vec![5.7,3.8,1.7,0.3,1.0],
        vec![5.1,3.8,1.5,0.3,1.0],
        vec![5.4,3.4,1.7,0.2,1.0],
        vec![5.1,3.7,1.5,0.4,1.0],
        vec![4.6,3.6,1.0,0.2,1.0],
        vec![5.1,3.3,1.7,0.5,1.0],
        vec![4.8,3.4,1.9,0.2,1.0],
        vec![5.0,3.0,1.6,0.2,1.0],
        vec![5.0,3.4,1.6,0.4,1.0],
        vec![5.2,3.5,1.5,0.2,1.0],
        vec![5.2,3.4,1.4,0.2,1.0],
        vec![4.7,3.2,1.6,0.2,1.0],
        vec![4.8,3.1,1.6,0.2,1.0],
        vec![5.4,3.4,1.5,0.4,1.0],
        vec![5.2,4.1,1.5,0.1,1.0],
        vec![5.5,4.2,1.4,0.2,1.0],
        vec![4.9,3.1,1.5,0.1,1.0],
        vec![5.0,3.2,1.2,0.2,1.0],
        vec![5.5,3.5,1.3,0.2,1.0],
        vec![4.9,3.1,1.5,0.1,1.0],
        vec![4.4,3.0,1.3,0.2,1.0],
        vec![5.1,3.4,1.5,0.2,1.0],
        vec![5.0,3.5,1.3,0.3,1.0],
        vec![4.5,2.3,1.3,0.3,1.0],
        vec![4.4,3.2,1.3,0.2,1.0],
        vec![5.0,3.5,1.6,0.6,1.0],
        vec![5.1,3.8,1.9,0.4,1.0],
        vec![4.8,3.0,1.4,0.3,1.0],
        vec![5.1,3.8,1.6,0.2,1.0],
        vec![4.6,3.2,1.4,0.2,1.0],
        vec![5.3,3.7,1.5,0.2,1.0],
        vec![5.0,3.3,1.4,0.2,1.0],
        vec![7.0,3.2,4.7,1.4,2.0],
        vec![6.4,3.2,4.5,1.5,2.0],
        vec![6.9,3.1,4.9,1.5,2.0],
        vec![5.5,2.3,4.0,1.3,2.0],
        vec![6.5,2.8,4.6,1.5,2.0],
        vec![5.7,2.8,4.5,1.3,2.0],
        vec![6.3,3.3,4.7,1.6,2.0],
        vec![4.9,2.4,3.3,1.0,2.0],
        vec![6.6,2.9,4.6,1.3,2.0],
        vec![5.2,2.7,3.9,1.4,2.0],
        vec![5.0,2.0,3.5,1.0,2.0],
        vec![5.9,3.0,4.2,1.5,2.0],
        vec![6.0,2.2,4.0,1.0,2.0],
        vec![6.1,2.9,4.7,1.4,2.0],
        vec![5.6,2.9,3.6,1.3,2.0],
        vec![6.7,3.1,4.4,1.4,2.0],
        vec![5.6,3.0,4.5,1.5,2.0],
        vec![5.8,2.7,4.1,1.0,2.0],
        vec![6.2,2.2,4.5,1.5,2.0],
        vec![5.6,2.5,3.9,1.1,2.0],
        vec![5.9,3.2,4.8,1.8,2.0],
        vec![6.1,2.8,4.0,1.3,2.0],
        vec![6.3,2.5,4.9,1.5,2.0],
        vec![6.1,2.8,4.7,1.2,2.0],
        vec![6.4,2.9,4.3,1.3,2.0],
        vec![6.6,3.0,4.4,1.4,2.0],
        vec![6.8,2.8,4.8,1.4,2.0],
        vec![6.7,3.0,5.0,1.7,2.0],
        vec![6.0,2.9,4.5,1.5,2.0],
        vec![5.7,2.6,3.5,1.0,2.0],
        vec![5.5,2.4,3.8,1.1,2.0],
        vec![5.5,2.4,3.7,1.0,2.0],
        vec![5.8,2.7,3.9,1.2,2.0],
        vec![6.0,2.7,5.1,1.6,2.0],
        vec![5.4,3.0,4.5,1.5,2.0],
        vec![6.0,3.4,4.5,1.6,2.0],
        vec![6.7,3.1,4.7,1.5,2.0],
        vec![6.3,2.3,4.4,1.3,2.0],
        vec![5.6,3.0,4.1,1.3,2.0],
        vec![5.5,2.5,4.0,1.3,2.0],
        vec![5.5,2.6,4.4,1.2,2.0],
        vec![6.1,3.0,4.6,1.4,2.0],
        vec![5.8,2.6,4.0,1.2,2.0],
        vec![5.0,2.3,3.3,1.0,2.0],
        vec![5.6,2.7,4.2,1.3,2.0],
        vec![5.7,3.0,4.2,1.2,2.0],
        vec![5.7,2.9,4.2,1.3,2.0],
        vec![6.2,2.9,4.3,1.3,2.0],
        vec![5.1,2.5,3.0,1.1,2.0],
        vec![5.7,2.8,4.1,1.3,2.0],
        vec![6.3,3.3,6.0,2.5,3.0],
        vec![5.8,2.7,5.1,1.9,3.0],
        vec![7.1,3.0,5.9,2.1,3.0],
        vec![6.3,2.9,5.6,1.8,3.0],
        vec![6.5,3.0,5.8,2.2,3.0],
        vec![7.6,3.0,6.6,2.1,3.0],
        vec![4.9,2.5,4.5,1.7,3.0],
        vec![7.3,2.9,6.3,1.8,3.0],
        vec![6.7,2.5,5.8,1.8,3.0],
        vec![7.2,3.6,6.1,2.5,3.0],
        vec![6.5,3.2,5.1,2.0,3.0],
        vec![6.4,2.7,5.3,1.9,3.0],
        vec![6.8,3.0,5.5,2.1,3.0],
        vec![5.7,2.5,5.0,2.0,3.0],
        vec![5.8,2.8,5.1,2.4,3.0],
        vec![6.4,3.2,5.3,2.3,3.0],
        vec![6.5,3.0,5.5,1.8,3.0],
        vec![7.7,3.8,6.7,2.2,3.0],
        vec![7.7,2.6,6.9,2.3,3.0],
        vec![6.0,2.2,5.0,1.5,3.0],
        vec![6.9,3.2,5.7,2.3,3.0],
        vec![5.6,2.8,4.9,2.0,3.0],
        vec![7.7,2.8,6.7,2.0,3.0],
        vec![6.3,2.7,4.9,1.8,3.0],
        vec![6.7,3.3,5.7,2.1,3.0],
        vec![7.2,3.2,6.0,1.8,3.0],
        vec![6.2,2.8,4.8,1.8,3.0],
        vec![6.1,3.0,4.9,1.8,3.0],
        vec![6.4,2.8,5.6,2.1,3.0],
        vec![7.2,3.0,5.8,1.6,3.0],
        vec![7.4,2.8,6.1,1.9,3.0],
        vec![7.9,3.8,6.4,2.0,3.0],
        vec![6.4,2.8,5.6,2.2,3.0],
        vec![6.3,2.8,5.1,1.5,3.0],
        vec![6.1,2.6,5.6,1.4,3.0],
        vec![7.7,3.0,6.1,2.3,3.0],
        vec![6.3,3.4,5.6,2.4,3.0],
        vec![6.4,3.1,5.5,1.8,3.0],
        vec![6.0,3.0,4.8,1.8,3.0],
        vec![6.9,3.1,5.4,2.1,3.0],
        vec![6.7,3.1,5.6,2.4,3.0],
        vec![6.9,3.1,5.1,2.3,3.0],
        vec![5.8,2.7,5.1,1.9,3.0],
        vec![6.8,3.2,5.9,2.3,3.0],
        vec![6.7,3.3,5.7,2.5,3.0],
        vec![6.7,3.0,5.2,2.3,3.0],
        vec![6.3,2.5,5.0,1.9,3.0],
        vec![6.5,3.0,5.2,2.0,3.0],
        vec![6.2,3.4,5.4,2.3,3.0],
        vec![5.9,3.0,5.1,1.8,3.0],
    ];


    let split = calc_split(&mut iris_data, 4);
    println!("{:?}", split);
}