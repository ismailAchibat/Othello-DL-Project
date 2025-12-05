import re
import pandas as pd
import matplotlib.pyplot as plt

def parse_training_log(log_file_path):

    # Try different encodings including UTF-16
    encodings = ['utf-16-le', 'utf-16', 'utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
    content = None
    
    for encoding in encodings:
        try:
            with open(log_file_path, 'r', encoding=encoding) as f:
                content = f.read()
            print(f"✓ Successfully read file with encoding: {encoding}")
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    
    if content is None:
        raise ValueError("Could not read file with any standard encoding. File might be corrupted.")
    
    # Extract basic info
    params_match = re.search(r'Number of parameters:\s*(\d+)', content)
    train_samples = re.search(r'Training.*?Number of samples\s*:\s*(\d+)', content, re.DOTALL)
    dev_samples = re.search(r'Development.*?Number of samples\s*:\s*(\d+)', content, re.DOTALL)
    
    # Extract epoch data - matching the exact format from your log
    # Pattern: epoch : 1/200 - loss = 4.119717276557374
    #          Accuracy Train:9.62%, Dev:9.04% ; Time:5 (last_train:2sec,
    epoch_pattern = r'epoch\s*:\s*(\d+)/\d+\s*-\s*loss\s*=\s*([\d.]+)\s+Accuracy\s+Train:([\d.]+)%,\s*Dev:([\d.]+)%\s*;\s*Time:(\d+)'
    epochs_data = re.findall(epoch_pattern, content)
    
    if not epochs_data:
        print("Trying alternative pattern with newline...")
        # Try with explicit newline
        epoch_pattern = r'epoch\s*:\s*(\d+)/\d+\s*-\s*loss\s*=\s*([\d.]+)\s*\nAccuracy\s+Train:([\d.]+)%,\s*Dev:([\d.]+)%\s*;\s*Time:(\d+)'
        epochs_data = re.findall(epoch_pattern, content)
    
    if not epochs_data:
        print("Trying most flexible pattern...")
        # Most flexible pattern
        epoch_pattern = r'epoch\s*:\s*(\d+)/\d+.*?loss\s*=\s*([\d.]+).*?Train:([\d.]+)%.*?Dev:([\d.]+)%.*?Time:(\d+)'
        epochs_data = re.findall(epoch_pattern, content, re.DOTALL)
    
    if not epochs_data:
        print("ERROR: Could not parse. Showing sample of file:")
        lines = content.split('\n')
        for i, line in enumerate(lines[:20]):
            print(f"{i}: {repr(line)}")
        raise ValueError("Could not parse log file. Please check the format.")
    
    print(f"✓ Successfully parsed {len(epochs_data)} epochs")
    
    # Extract best scores
    best_pattern = r'\*+\s*The best score on DEV\s+(\d+)\s*:([\d.]+)%'
    best_scores = re.findall(best_pattern, content)
    print(f"✓ Found {len(best_scores)} best score markers")
    
    # Create DataFrame
    df = pd.DataFrame(epochs_data, columns=['Epoch', 'Loss', 'Train_Acc', 'Dev_Acc', 'Time'])
    df = df.astype({
        'Epoch': int,
        'Loss': float,
        'Train_Acc': float,
        'Dev_Acc': float,
        'Time': int
    })
    
    # Add best score indicator
    best_epochs = set(int(epoch) for epoch, _ in best_scores)
    df['Is_Best'] = df['Epoch'].isin(best_epochs)
    
    # Create summary dictionary
    summary = {
        'num_parameters': int(params_match.group(1)) if params_match else None,
        'train_samples': int(train_samples.group(1)) if train_samples else None,
        'dev_samples': int(dev_samples.group(1)) if dev_samples else None,
        'total_epochs': len(df),
        'best_dev_acc': df['Dev_Acc'].max(),
        'best_epoch': int(df.loc[df['Dev_Acc'].idxmax(), 'Epoch']),
        'final_train_acc': df.iloc[-1]['Train_Acc'],
        'final_dev_acc': df.iloc[-1]['Dev_Acc'],
        'total_time_sec': int(df.iloc[-1]['Time']),
        'initial_dev_acc': df.iloc[0]['Dev_Acc']
    }
    
    return df, summary


def generate_summary_table(summary):
    """Generate a formatted summary table."""
    summary_df = pd.DataFrame([
        ['Model Parameters', f"{summary['num_parameters']:,}"],
        ['Training Samples', f"{summary['train_samples']:,}"],
        ['Dev Samples', f"{summary['dev_samples']:,}"],
        ['Total Epochs', summary['total_epochs']],
        ['', ''],
        ['Initial Dev Accuracy', f"{summary['initial_dev_acc']:.2f}%"],
        ['Best Dev Accuracy', f"{summary['best_dev_acc']:.2f}%"],
        ['Best Epoch', summary['best_epoch']],
        ['Final Train Accuracy', f"{summary['final_train_acc']:.2f}%"],
        ['Final Dev Accuracy', f"{summary['final_dev_acc']:.2f}%"],
        ['', ''],
        ['Total Training Time', f"{summary['total_time_sec']} sec ({summary['total_time_sec']/60:.1f} min)"],
        ['Improvement', f"{summary['best_dev_acc'] - summary['initial_dev_acc']:.2f}%"]
    ], columns=['Metric', 'Value'])
    
    return summary_df


def generate_milestone_table(df, milestones=[1, 10, 50, 100, 150, 200]):
    """Generate table showing progress at key milestones."""
    milestone_data = []
    for epoch in milestones:
        if epoch <= len(df):
            row = df[df['Epoch'] == epoch].iloc[0]
            milestone_data.append({
                'Epoch': epoch,
                'Loss': f"{row['Loss']:.4f}",
                'Train Acc (%)': f"{row['Train_Acc']:.2f}",
                'Dev Acc (%)': f"{row['Dev_Acc']:.2f}",
                'Time (sec)': row['Time'],
                'Best?': '✓' if row['Is_Best'] else ''
            })
    
    return pd.DataFrame(milestone_data)


def plot_training_curves(df, summary):
    """Create visualization of training progress."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Accuracy curves
    axes[0].plot(df['Epoch'], df['Train_Acc'], label='Train Accuracy', linewidth=2)
    axes[0].plot(df['Epoch'], df['Dev_Acc'], label='Dev Accuracy', linewidth=2)
    axes[0].axvline(x=summary['best_epoch'], color='r', linestyle='--', 
                    label=f'Best Epoch ({summary["best_epoch"]})', alpha=0.7)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].set_title('Training and Development Accuracy', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Loss curve
    axes[1].plot(df['Epoch'], df['Loss'], color='orange', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# Main execution
if __name__ == "__main__":
    import sys
    
    # Accept log file from command line or use default
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    else:
        log_file = "first_One2Oneoutput.log"
    
    print("="*60)
    print("TRAINING REPORT")
    print("="*60)
    print(f"📁 Reading log file: {log_file}\n")
    
    try:
        # Parse data
        df, summary = parse_training_log(log_file)
        
        # Generate and display summary
        print("\n📊 TRAINING SUMMARY")
        print("-"*60)
        summary_table = generate_summary_table(summary)
        print(summary_table.to_string(index=False))
        
        # Generate and display milestone table
        print("\n\n📈 KEY MILESTONES")
        print("-"*60)
        milestone_table = generate_milestone_table(df)
        print(milestone_table.to_string(index=False))
        
        # Save to CSV
        df.to_csv('results/second_optimized_cnn_output/training_results_full.csv', index=False)
        summary_table.to_csv('results/second_optimized_cnn_output/training_summary.csv', index=False)
        milestone_table.to_csv('results/second_optimized_cnn_output/training_milestones.csv', index=False)
        
        print("\n\n💾 Files saved:")
        print("  - results/second_optimized_cnn_output/training_results_full.csv")
        print("  - results/second_optimized_cnn_output/training_summary.csv")
        print("  - results/second_optimized_cnn_output/training_milestones.csv")
        print("  - results/second_optimized_cnn_output/training_curves.png")
        
        # Generate and save plots
        fig = plot_training_curves(df, summary)
        fig.savefig('results/second_optimized_cnn_output/training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\n✅ Report generation complete!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("\nPlease check that your log file exists and has the correct format.")
        sys.exit(1)


