#!/usr/bin/env python3

import pandas as pd
import os
import sys
import aiohttp
import asyncio
import tarfile
import shutil
from pathlib import Path
from tqdm.asyncio import tqdm
import argparse
import mimetypes
import time

class TokenBucket:
    """
    A token bucket implementation for rate limiting.
    """
    def __init__(self, rate, capacity):
        """
        Initialize the token bucket.
        :param rate: Number of tokens added per second.
        :param capacity: Maximum number of tokens the bucket can hold.
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_refill = time.time()

    async def acquire(self):
        """
        Wait until a token is available and consume it.
        """
        while True:
            self._refill()
            if self.tokens >= 1:
                self.tokens -= 1
                return
            await asyncio.sleep(0.01)  # Wait a short time before checking again

    def _refill(self):
        """
        Refill the bucket with tokens based on the elapsed time.
        """
        now = time.time()
        elapsed = now - self.last_refill
        new_tokens = elapsed * self.rate
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_refill = now

    def adjust_rate(self, new_rate):
        new_rate = max(10, new_rate)
        if new_rate != self.rate:
            print(f"[TokenBucket] Adjusting rate: {self.rate:.2f} -> {new_rate:.2f} tokens/sec")
            self.rate = new_rate

    def get_rate(self):
        return self.rate

def parse_args():
    """
    Parse user inputs from arguments using argparse.
    """
    parser = argparse.ArgumentParser(description="Download images asynchronously and tar the output folder.")
    
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input CSV or Parquet file.")
    parser.add_argument("--output_tar", type=str, required=True, help="Path to the output tar file (e.g., 'images.tar.gz').")
    parser.add_argument("--url_name", type=str, default="photo_url", help="Column name containing the image URLs.")
    parser.add_argument("--class_name", type=str, default="taxon_name", help="Column name containing the class names.")

    return parser.parse_args()


async def download_image_with_extensions(session, semaphore, row, output_folder, url_col, class_col, token_bucket):
    """Download an image asynchronously with retries for different file extensions and token bucket rate limiting."""
    # Extensions to try if the original extension fails
    fallback_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.pdf']
    async with semaphore:
        key, image_url = str(row.name), str(row[url_col])
        if row[class_col] is None:
            class_name = "unknown"
        else:
            class_name = str(row[class_col]).replace("'", "").replace(" ", "_")
        base_url, original_ext = os.path.splitext(image_url)

        # Wait for a token before proceeding
        await token_bucket.acquire()

        # Check if the URL has an extension
        if not original_ext:
            try:
                async with session.get(image_url) as response:
                    if response.status == 200:
                        content = await response.read()
                        mime_type = response.headers.get('Content-Type')
                        ext = mimetypes.guess_extension(mime_type)
                        if ext:
                            file_name = f"{base_url.split('/')[-1]}{ext}"
                            file_path = os.path.join(output_folder, class_name, file_name)
                            os.makedirs(os.path.dirname(file_path), exist_ok=True)
                            with open(file_path, 'wb') as f:
                                f.write(content)
                            return key, file_name, class_name, None, image_url
                        else:
                            a = 1
                            #print(f"Failed to determine extension for MIME type: {mime_type}")
                    elif response.status == 429:
                        return key, None, class_name, "429", image_url
                    else:
                        #print(f"Failed to download {image_url}: HTTP {response.status}")
                        error = response.status
            except Exception as err:
                #print(f"Error with URL {image_url}: {err}")
                return key, None, class_name, str(err), image_url
        else:
            file_name = f"{base_url.split('/')[-1]}{original_ext}"
            file_path = os.path.join(output_folder, class_name, file_name)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            #print(file_path)
            # Try downloading the image with the original extension first
            try:
                async with session.get(image_url) as response:
                    if response.status == 200:
                        content = await response.read()
                        with open(file_path, 'wb') as f:
                            f.write(content)
                        return key, file_name, class_name, None, image_url
                    elif response.status == 429:
                        return key, None, class_name, "429", image_url
                    else:
                        #print(f"Failed to download {file_name}: HTTP {response.status}")
                        error = response.status
            except Exception as err:
                a = 1
                #print(f"Error with original URL {image_url}: {err}")

            # If original extension fails, try the fallback extensions
            for ext in fallback_extensions:
                if ext == original_ext:  # Skip if the fallback extension is the same as the original
                    continue
                new_url = f"{base_url}{ext}"
                file_name = f"{base_url.split('/')[-1]}{ext}"
                file_path = os.path.join(output_folder, class_name, file_name)

                #print(file_path)
                try:
                    async with session.get(new_url) as response:
                        if response.status == 200:
                            content = await response.read()
                            with open(file_path, 'wb') as f:
                                f.write(content)
                            return key, file_name, class_name, None, image_url
                        elif response.status == 429:
                            return key, None, class_name, "429", image_url
                        else:
                            #print(f"Failed to download {file_name}: HTTP {response.status}")
                            error = response.status
                except Exception as err:
                    #print(f"Error with {new_url}: {err}")
                    continue  # Try the next extension

        # If all extensions fail
        return key, None, class_name, "All extensions failed.", image_url

async def slowly_increase_rate(token_bucket, max_rate, interval=5):
    while True:
        await asyncio.sleep(interval)
        current_rate = token_bucket.get_rate()
        if current_rate < max_rate:
            token_bucket.adjust_rate(current_rate * 1.2)

async def main():
    args = parse_args()
    input_path = args.input_path
    output_tar_path = args.output_tar
    url_col = args.url_name 
    class_col = args.class_name

    concurrent_downloads = 30
    initial_rate_limit = 30
    bucket_capacity = 60

    token_bucket = TokenBucket(rate=initial_rate_limit, capacity=bucket_capacity)
    max_safe_rate = initial_rate_limit

    output_folder = os.path.splitext(os.path.basename(output_tar_path))[0]
    if output_tar_path.endswith(".tar.gz"):
        output_folder = os.path.splitext(output_folder)[0]

    if input_path.endswith(".parquet"):
        df = pd.read_parquet(input_path)
    elif input_path.endswith(".csv"):
        df = pd.read_csv(input_path)
    else:
        print("Unsupported file format.")
        sys.exit(1)

    semaphore = asyncio.Semaphore(concurrent_downloads)
    retry_df = df.copy()
    attempt = 1

    async with aiohttp.ClientSession() as session:
        recovery_task = asyncio.create_task(slowly_increase_rate(token_bucket, max_safe_rate))

        while not retry_df.empty:
            print(f"\n Attempt #{attempt} ? {len(retry_df)} images at {token_bucket.get_rate():.2f} req/s")
            tasks = [
                asyncio.create_task(
                    download_image_with_extensions(session, semaphore, row, output_folder, url_col, class_col, token_bucket)
                )
                for _, row in retry_df.iterrows()
            ]

            failed_rows = []

            for future in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
                try:
                    key, file_name, class_name, error, image_url = await future
                    if error:
                        token_bucket.adjust_rate(token_bucket.get_rate() * 0.75)
                except Exception as e:
                    print(f"Error during task execution: {e}")
                    failed_rows.append(retry_df.loc[key])  
                    print(f"Failed to download {file_name} for {class_name}, url: {image_url}, err: {error}, add to retry list")
                    error = e

            if failed_rows :
                #print(f"{len(failed_rows)} failed. Retrying after cooldown...")
                retry_df = pd.DataFrame(failed_rows)
                attempt += 1
                await asyncio.sleep(0.1)
            else:
                print("All downloads succeeded!")
                break

        recovery_task.cancel()

    # with tarfile.open(output_tar_path, "w:gz") as tar:
    #     tar.add(output_folder, arcname=os.path.basename(output_folder))

    print(f"Downloaded and archived to: {Path(output_tar_path).resolve()}")


if __name__ == '__main__':
    asyncio.run(main())