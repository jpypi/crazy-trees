use std::cmp::Eq;
use std::hash::Hash;
use std::collections::HashMap;
use std::io::prelude::*;
use std::io::BufReader;
use std::fs::{File};

type Num = f32;
type Sample = Vec<Num>;


#[derive(Debug)]
struct Split {
    feature: usize,
    point: Num,
    gain: Num,
}

impl Split {
    fn new() -> Self {
        Split {
            feature: 0,
            point: 0.0,
            gain: 0.0,
        }
    }
}



#[derive(Debug)]
struct TreeNode {
    pub split: Split,
    pub left_child: Option<Box<TreeNode>>,
    pub right_child: Option<Box<TreeNode>>,
}

impl TreeNode {
    fn new(split: Split) -> Self {
        TreeNode {
            split,
            left_child: None,
            right_child: None,
        }
    }
}

fn show_tree(root: &Option<Box<TreeNode>>, depth: u32) -> String {
    let indent = ((depth + 1) * 4) as usize;

    if let &Some(ref root) = root {
        format!("TreeNode {{
{0:1$}split: {3:?}
{0:1$}left_child:  {4:}
{0:1$}right_child: {5:}
{0:2$}}}", "", indent, indent - 4, root.split,
                show_tree(&root.left_child, depth + 1),
                show_tree(&root.right_child, depth + 1))
    } else {
        "None".to_string()
    }
}

/*
struct Dataset<'a> {
    label_i: usize,
    data: Vec<Sample<'a>>,
}
*/


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


#[allow(dead_code)]
fn gini(dist: &Vec<Num>) -> Num {
    dist.iter().fold(1.0, |acc, &x| acc - x*x)
}


fn entropy(dist: &Vec<Num>) -> Num {
    -dist.iter().fold(0.0, |acc, &x| acc + x * x.log2())
}


fn get_slice_entropy(data: &[&Sample], label_i: usize) -> Num {
    let labels = data.iter().map(|x| x[label_i] as i32).collect();
    entropy(&count_dist(&labels))
}


fn feature_split(data: &mut Vec<&Sample>, feature_i: usize, label_i: usize) -> (Num, Num) {
    // Sort the data by the feature
    data.sort_by(|a, b| a[feature_i].partial_cmp(&b[feature_i]).unwrap());
    //println!("{:?}\n~~~~~~~~~~~~~~~~~~~~~", data);

    // Calculate the current entropy before any split
    let parent_entropy = get_slice_entropy(data, label_i);

    let (mut best_split_point, mut best_info_gain) = (0.0, 0.0);

    for i in 0..(data.len() - 1) {
        let split_point = (data[i][feature_i] + data[i+1][feature_i]) / 2.0;

        let left  = &data[..i+1];
        let right = &data[i+1..];

        let left_entropy = get_slice_entropy(left, label_i);
        let right_entropy = get_slice_entropy(right, label_i);

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


fn calc_split(data: &mut Vec<&Sample>, label_i: usize) -> Split {
    let mut split = Split::new();

    // Loop through features to find the best feature to use
    for i in 0..data[0].len() {
        if i != label_i {
            let (ig, sp) = feature_split(data, i, label_i);
            //println!("Split on feature: {} | Information gain: {} | Split point: {}",i, ig, sp);
            if ig > split.gain {
                split.feature = i;
                split.point = sp;
                split.gain = ig;
            }
        }
    }

    split
}


fn fit_tree(data: &mut Vec<&Sample>, label_i: usize, depth: i32) -> Option<Box<TreeNode>> {
    if data.len() <= 6 {
        return None;
    }

    println!("\ndepth: {} | data size: {}", depth, data.len());
    let mut cur = Box::new(TreeNode::new(calc_split(data, label_i)));
    println!("{:?}", cur.split);

    if cur.split.gain > 0.0 {
        let mut left: Vec<&Sample> = data.iter()
                                         .filter(|x| x[cur.split.feature] < cur.split.point)
                                         .map(|x| *x).collect();
        let mut right: Vec<&Sample> = data.iter()
                                          .filter(|x| x[cur.split.feature] >= cur.split.point)
                                          .map(|x| *x).collect();

        cur.left_child = fit_tree(&mut left, label_i, depth + 1);
        cur.right_child = fit_tree(&mut right, label_i, depth + 1);

        return Some(cur);
    } else {
        return None;
    }
}



fn load_data(filename: &str) -> Result<Vec<Vec<f32>>, std::io::Error> {
    let mut data = Vec::new();

    for ln in BufReader::new(File::open(filename)?).lines() {
        if let Ok(line) = ln {
            data.push(line.split(",")
                          .map( |l| l.trim().parse().unwrap())
                          .collect());
        }
    }

    Ok(data)
}

fn main() {
    let temp_iris_data = load_data("iris.csv");
    if let Ok(iris_data) = temp_iris_data {
        let mut id = iris_data.iter().map(|e| e).collect();
        let t = fit_tree(&mut id, 4, 0);
        println!("{}", show_tree(&Some(t.unwrap()), 0));
    }
}
