import torch
import pandas as pd
import pickle
from torch_geometric.data import Data

class MovieLensDataset:
    """
    MovieLens Dataset Loader for GNN
    """
    def __init__(self, data_dir='../data'):
        self.data_dir = data_dir
        self.load_data()
    
    def load_data(self):
        """
        Load all dataset components
        """
        # Load graph structure
        graph_data = torch.load(f'{self.data_dir}/graph_data.pt')
        
        self.x = graph_data['x']
        self.edge_index = graph_data['edge_index']
        self.edge_attr = graph_data['edge_attr']
        self.node_type = graph_data['node_type']
        self.num_users = graph_data['num_users']
        self.num_movies = graph_data['num_movies']
        self.num_nodes = graph_data['num_nodes']
        self.num_features = graph_data['num_features']
        
        # Load metadata
        with open(f'{self.data_dir}/metadata.pkl', 'rb') as f:
            self.metadata = pickle.load(f)
        
        # Load train/test splits
        self.train_ratings = pd.read_csv(f'{self.data_dir}/train_ratings.csv')
        self.test_pairs = pd.read_csv(f'{self.data_dir}/test_pairs.csv')
        
        # Prepare train data for PyTorch
        self._prepare_train_data()
        self._prepare_test_data()
    
    def _prepare_train_data(self):
        """
        Prepare training edges and labels
        """
        # Convert train ratings to tensor indices
        user_idx = torch.tensor(self.train_ratings['user_id'].values - 1, dtype=torch.long)
        movie_idx = torch.tensor(self.train_ratings['movie_id'].values - 1 + self.num_users, 
                                dtype=torch.long)
        ratings = torch.tensor(self.train_ratings['rating'].values, dtype=torch.float)
        
        self.train_user_idx = user_idx
        self.train_movie_idx = movie_idx
        self.train_ratings_tensor = ratings
    
    def _prepare_test_data(self):
        """
        Prepare test edges for prediction
        """
        # Convert test pairs to tensor indices
        user_idx = torch.tensor(self.test_pairs['user_id'].values - 1, dtype=torch.long)
        movie_idx = torch.tensor(self.test_pairs['movie_id'].values - 1 + self.num_users,
                                dtype=torch.long)
        
        self.test_user_idx = user_idx
        self.test_movie_idx = movie_idx
    
    def get_pyg_data(self):
        """
        Return PyTorch Geometric Data object
        """
        data = Data(
            x=self.x,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            node_type=self.node_type,
            num_users=self.num_users,
            num_movies=self.num_movies,
            num_nodes=self.num_nodes
        )
        
        # Add train data
        data.train_user_idx = self.train_user_idx
        data.train_movie_idx = self.train_movie_idx
        data.train_ratings = self.train_ratings_tensor
        
        # Add test data
        data.test_user_idx = self.test_user_idx
        data.test_movie_idx = self.test_movie_idx
        
        return data
    
    def get_train_batch(self, batch_size=1024):
        """
        Get a random batch of training edges
        """
        num_train = len(self.train_user_idx)
        indices = torch.randperm(num_train)[:batch_size]
        
        return (
            self.train_user_idx[indices],
            self.train_movie_idx[indices],
            self.train_ratings_tensor[indices]
        )
    
    def print_statistics(self):
        """
        Print dataset statistics
        """
        print("=" * 60)
        print("DATASET STATISTICS")
        print("=" * 60)
        print(f"Nodes:")
        print(f"  Users:        {self.num_users}")
        print(f"  Movies:       {self.num_movies}")
        print(f"  Total:        {self.num_nodes}")
        print(f"\nFeatures:")
        print(f"  Dimension:    {self.num_features}")
        print(f"\nEdges:")
        print(f"  Train:        {len(self.train_user_idx)}")
        print(f"  Test:         {len(self.test_user_idx)}")
        print(f"  Graph edges:  {self.edge_index.shape[1]} (bidirectional)")
        print(f"\nRatings:")
        print(f"  Min:          {self.metadata['rating_min']}")
        print(f"  Max:          {self.metadata['rating_max']}")
        print("=" * 60)

# Convenience function
def load_movielens_data(data_dir='../data'):
    """
    Load MovieLens dataset
    
    Returns:
        data: PyTorch Geometric Data object
        dataset: MovieLensDataset object with helper methods
    """
    dataset = MovieLensDataset(data_dir)
    data = dataset.get_pyg_data()
    return data, dataset

if __name__ == "__main__":
    data, dataset = load_movielens_data()
    dataset.print_statistics()
    print("\n✅ Data loaded successfully!")