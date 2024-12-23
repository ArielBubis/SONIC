import asyncio
import libtorrent as lt

def download_torrent(torrent_file, output_dir):
    # Create session
    session = lt.session()
    session.listen_on(6881, 6891)
    
    # if is file path
    if torrent_file.endswith('.torrent'):
        # Add torrent
        info = lt.torrent_info(torrent_file)
        handle = session.add_torrent({"ti": info, "save_path": output_dir})
    elif torrent_file.startswith('magnet:'):
        # Add magnet link
        handle = lt.add_magnet_uri(session, torrent_file, {"save_path": output_dir})
    
    # Download torrent
    while not handle.is_seed():
        s = handle.status()
        print(f"Progress: {s.progress * 100:.2f}%")
        asyncio.sleep(1)
    
    print("Download complete!")