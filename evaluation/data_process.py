import os
import zipfile
import tarfile
import tempfile
import shutil
from tqdm import tqdm
from datetime import datetime, timedelta
from utilities.util import get_config


def configured_raw_root():
    return get_config().get("paths", {}).get("raw_archive_root", "./raw_data")

def sanitize_path(path):
    """
    Replace problematic characters in path components.

    Example:
      SR-1 : (224) Hawthorne Blvd
      ->
      SR-1 - (224) Hawthorne Blvd
    """

    parts = path.split("/")

    clean_parts = []

    for part in parts:

        # Replace colon with dash
        part = part.replace(":", "-")

        # Optional cleanup
        part = part.strip()

        clean_parts.append(part)

    return os.path.join(*clean_parts)


def unzip_pulled_data_files(folder_path):
    """
    Extract ZIP files like:

        pulled_data_xxx.zip

    which internally contain:

        pulled_data/sensor_name/some_date/...

    Instead of extracting files directly, each
    `some_date` folder becomes:

        some_date.tar

    Example result:

        pulled_data_xxx/
            pulled_data/
                cctv/
                    20250102.tar

    Uses a temporary directory INSIDE folder_path
    instead of /tmp.
    """

    zip_files = [
        f for f in os.listdir(folder_path)
        if f.startswith("pulled_data")
        and f.endswith(".zip")
    ]

    total_zips = len(zip_files)

    for zip_index, filename in enumerate(zip_files, start=1):

        zip_path = os.path.join(folder_path, filename)

        extract_folder_name = os.path.splitext(filename)[0]

        extract_folder_path = os.path.join(
            folder_path,
            extract_folder_name
        )

        os.makedirs(extract_folder_path, exist_ok=True)

        print(f"\n[{zip_index}/{total_zips}] Processing: {filename}")

        try:

            #
            # TEMP DIRECTORY INSIDE folder_path
            #
            # temp_extract = tempfile.mkdtemp(
            #     prefix="temp_extract_",
            #     dir=folder_path
            # )
            temp_extract = tempfile.mkdtemp()

            print(f"Temporary extraction dir:")
            print(f"  {temp_extract}")

            with zipfile.ZipFile(zip_path, "r") as zip_ref:

                members = zip_ref.infolist()
                total_files = len(members)

                #
                # Extract into temp directory
                #
                for file_index, member in enumerate(members, start=1):

                    original_name = member.filename

                    clean_name = sanitize_path(
                        original_name
                    )

                    output_path = os.path.join(
                        temp_extract,
                        clean_name
                    )

                    if member.is_dir():

                        os.makedirs(
                            output_path,
                            exist_ok=True
                        )

                    else:

                        os.makedirs(
                            os.path.dirname(output_path),
                            exist_ok=True
                        )

                        with zip_ref.open(member) as source:
                            with open(output_path, "wb") as target:
                                shutil.copyfileobj(
                                    source,
                                    target
                                )

                    percent = (
                        file_index / total_files
                    ) * 100

                    print(
                        f"  Extracting "
                        f"[{file_index}/{total_files}] "
                        f"({percent:.1f}%) "
                        f"{clean_name}"
                    )

            #
            # Convert each some_date folder into TAR
            #

            pulled_data_root = os.path.join(
                temp_extract,
                "pulled_data"
            )

            if not os.path.exists(pulled_data_root):
                print("No pulled_data folder found.")
                continue

            sensors = os.listdir(pulled_data_root)

            for sensor_name in sensors:

                sensor_path = os.path.join(
                    pulled_data_root,
                    sensor_name
                )

                if not os.path.isdir(sensor_path):
                    continue

                final_sensor_folder = os.path.join(
                    extract_folder_path,
                    "pulled_data",
                    sensor_name
                )

                os.makedirs(
                    final_sensor_folder,
                    exist_ok=True
                )

                date_folders = os.listdir(sensor_path)

                for date_folder in date_folders:

                    date_path = os.path.join(
                        sensor_path,
                        date_folder
                    )

                    if not os.path.isdir(date_path):
                        continue

                    tar_output_path = os.path.join(
                        final_sensor_folder,
                        f"{date_folder}.tar"
                    )

                    print(
                        f"  Creating TAR: "
                        f"{tar_output_path}"
                    )

                    with tarfile.open(
                        tar_output_path,
                        "w"
                    ) as tar:

                        tar.add(
                            date_path,
                            arcname=date_folder
                        )

            #
            # CLEANUP TEMP DIRECTORY
            #

            print(
                f"Removing temp directory:"
                f" {temp_extract}"
            )

            shutil.rmtree(temp_extract)

            print(f"Finished: {filename}")

            new_filename = f"_{filename}"

            new_zip_path = os.path.join(
                folder_path,
                new_filename
            )

            print(f"Renaming ZIP:")
            print(f"  FROM: {zip_path}")
            print(f"  TO  : {new_zip_path}")

            os.rename(
                zip_path,
                new_zip_path
            )

        except zipfile.BadZipFile:

            print(f"Invalid ZIP file: {filename}")

        except Exception as e:

            print(f"Error processing {filename}: {e}")



def move_tar_files_to_backup(
    source_root,
    backup_root="sigmus_backup/raw"
):
    """
    Traverse ONLY folders beginning with:

        pulled_data*

    Handles BOTH:

        .../pulled_data/sensor/20250102.tar

    AND:

        .../pulled_data/sensor/20250102/

    Folder dates are first converted into TARs.

    Final result:

        sigmus_backup/raw/sensor/20250102.tar

    Existing TARs are skipped.
    """

    #
    # First collect all candidate items
    #
    work_items = []

    top_level_entries = os.listdir(source_root)

    pulled_data_folders = [
        entry for entry in top_level_entries
        if entry.startswith("pulled_data")
        and os.path.isdir(
            os.path.join(source_root, entry)
        )
    ]

    print("Creating work items...")
    for pulled_folder in tqdm(pulled_data_folders):

        pulled_folder_path = os.path.join(
            source_root,
            pulled_folder
        )
        print(f"looking in {pulled_folder}")

        for root, dirs, files in os.walk(
            pulled_folder_path
        ):

            #
            # Existing TAR files
            #
            for file in files:

                if file.endswith(".tar"):

                    work_items.append({
                        "type": "tar",
                        "path": os.path.join(root, file)
                    })

            #
            # Date folders
            #
            for d in dirs:

                #
                # Match YYYYMMDD folders
                #
                if (
                    len(d) == 8
                    and d.isdigit()
                ):

                    work_items.append({
                        "type": "folder",
                        "path": os.path.join(root, d)
                    })

    moved_count = 0
    skipped_count = 0

    #
    # Progress bar
    #
    for item in tqdm(
        work_items,
        desc="Processing TARs/Folders"
    ):

        item_type = item["type"]
        item_path = item["path"]

        try:

            #
            # Determine sensor name
            #
            path_parts = item_path.split(os.sep)

            pulled_data_index = path_parts.index(
                "pulled_data"
            )

            sensor_name = path_parts[
                pulled_data_index + 1
            ]

            #
            # Destination sensor folder
            #
            dest_sensor_folder = os.path.join(
                backup_root,
                sensor_name
            )

            os.makedirs(
                dest_sensor_folder,
                exist_ok=True
            )

            #
            # CASE 1:
            # Existing TAR file
            #
            if item_type == "tar":

                tar_filename = os.path.basename(
                    item_path
                )

                dest_tar_path = os.path.join(
                    dest_sensor_folder,
                    tar_filename
                )

                #
                # Skip existing
                #
                if os.path.exists(dest_tar_path):

                    skipped_count += 1
                    continue

                # print(f"Moving from {item_path} to {dest_tar_path}")
                shutil.move(
                    item_path,
                    dest_tar_path
                )

                moved_count += 1

            #
            # CASE 2:
            # Date folder -> convert to TAR
            #
            elif item_type == "folder":

                date_folder = os.path.basename(
                    item_path
                )

                tar_filename = f"{date_folder}.tar"

                dest_tar_path = os.path.join(
                    dest_sensor_folder,
                    tar_filename
                )

                #
                # Skip existing
                #
                if os.path.exists(dest_tar_path):

                    skipped_count += 1
                    continue


                # print(f"Moving from {item_path} to {dest_tar_path}")
                # Create TAR directly at destination
                
                with tarfile.open(
                    dest_tar_path,
                    "w"
                ) as tar:

                    tar.add(
                        item_path,
                        arcname=date_folder
                    )

                moved_count += 1

        except Exception as e:

            print(
                f"\nError processing "
                f"{item_path}: {e}"
            )

    print("\nDone.")
    print(f"Moved/Created : {moved_count}")
    print(f"Skipped       : {skipped_count}")


def find_missing_sensor_dates(raw_root=None):
    """
    For each sensor folder inside:

        sigmus_backup/raw/<sensor_name>/

    inspect all YYYYMMDD.tar files and identify
    missing dates in the sequence.
    """

    raw_root = raw_root or configured_raw_root()
    sensor_folders = [
        d for d in os.listdir(raw_root)
        if os.path.isdir(
            os.path.join(raw_root, d)
        )
    ]

    all_results = {}

    for sensor_name in sorted(sensor_folders):

        sensor_path = os.path.join(
            raw_root,
            sensor_name
        )

        #
        # Collect valid dates
        #
        dates = []

        for file in os.listdir(sensor_path):

            if not file.endswith(".tar"):
                continue

            #
            # Remove .tar
            #
            date_str = os.path.splitext(file)[0]

            #
            # Ensure YYYYMMDD format
            #
            if (
                len(date_str) == 8
                and date_str.isdigit()
            ):

                try:

                    dt = datetime.strptime(
                        date_str,
                        "%Y%m%d"
                    )

                    dates.append(dt)

                except ValueError:
                    pass

        #
        # Skip empty sensors
        #
        if not dates:
            continue

        dates = sorted(set(dates))

        start_date = dates[0]
        end_date = dates[-1]

        existing_dates = set(dates)

        missing_dates = []

        current = start_date

        while current <= end_date:

            if current not in existing_dates:

                missing_dates.append(
                    current.strftime("%Y%m%d")
                )

            current += timedelta(days=1)

        all_results[sensor_name] = {
            "start_date": start_date.strftime(
                "%Y%m%d"
            ),
            "end_date": end_date.strftime(
                "%Y%m%d"
            ),
            "total_dates": len(dates),
            "missing_dates": missing_dates
        }

    #
    # Print summary
    #
    for sensor_name, result in all_results.items():

        print("\n===================================")
        print(f"Sensor: {sensor_name}")
        print(
            f"Date Range: "
            f"{result['start_date']} "
            f"-> "
            f"{result['end_date']}"
        )

        print(
            f"Total Existing Dates: "
            f"{result['total_dates']}"
        )

        if result["missing_dates"]:

            print(
                f"Missing Dates "
                f"({len(result['missing_dates'])}):"
            )

            print(
                ", ".join(
                    result["missing_dates"]
                )
            )

        else:

            print("No missing dates.")
        
        input("Press enter to continue...")

    return all_results


def _is_safe_member_path(base_dir, member_name):
    """
    Return True if member_name would extract inside base_dir.
    """
    base_dir_abs = os.path.abspath(base_dir)
    target_path_abs = os.path.abspath(
        os.path.join(base_dir_abs, member_name)
    )

    return (
        target_path_abs == base_dir_abs
        or target_path_abs.startswith(base_dir_abs + os.sep)
    )


def _is_safe_link_target(base_dir, member):
    """
    Return True if a tar symlink/hardlink target stays inside base_dir.
    """
    if not (
        member.issym()
        or member.islnk()
    ):
        return True

    if os.path.isabs(member.linkname):
        return False

    link_parent = os.path.dirname(member.name)

    link_target_path = os.path.join(
        base_dir,
        link_parent,
        member.linkname
    )

    base_dir_abs = os.path.abspath(base_dir)
    link_target_abs = os.path.abspath(link_target_path)

    return (
        link_target_abs == base_dir_abs
        or link_target_abs.startswith(base_dir_abs + os.sep)
    )


def safe_extract_tar(
    tar_path,
    extract_dir
):
    """
    Safely extract a TAR file into extract_dir.

    This prevents path traversal from entries such as:

        ../../somewhere_else/file.txt

    and rejects absolute or escaping symlink/hardlink targets.
    """
    os.makedirs(
        extract_dir,
        exist_ok=True
    )

    with tarfile.open(
        tar_path,
        "r:*"
    ) as tar:

        members = tar.getmembers()

        for member in members:

            if not _is_safe_member_path(
                extract_dir,
                member.name
            ):
                raise RuntimeError(
                    f"Unsafe TAR path in {tar_path}: {member.name}"
                )

            if not _is_safe_link_target(
                extract_dir,
                member
            ):
                raise RuntimeError(
                    f"Unsafe TAR link in {tar_path}: "
                    f"{member.name} -> {member.linkname}"
                )

        tar.extractall(
            extract_dir,
            members=members
        )


def tar_has_date_top_folder(
    tar_path,
    date_str
):
    """
    Return True if every non-empty TAR member is inside date_str/.

    This handles TARs created with:

        tar.add(date_path, arcname=date_folder)

    where the archive layout is:

        20250102/...
    """
    with tarfile.open(
        tar_path,
        "r:*"
    ) as tar:

        members = [
            member for member in tar.getmembers()
            if member.name.strip("/")
        ]

        if not members:
            return False

        for member in members:

            first_part = member.name.strip("/").split("/")[0]

            if first_part != date_str:
                return False

    return True


def temp_source_date_has_extracted_data(
    temp_root,
    data_source,
    date_str
):
    """
    Return True if temp_root/data_source/date_str already contains extracted data.

    This is intentionally conservative: if the folder exists and contains any
    file, we treat it as usable and do not re-copy/re-untar over it.
    """
    date_temp_folder = os.path.join(
        temp_root,
        data_source,
        date_str
    )

    if not os.path.isdir(date_temp_folder):
        return False

    for _root, _dirs, files in os.walk(date_temp_folder):
        if files:
            return True

    return False


def reset_temp_folder(
    temp_root="./evaluation/temp",
    clear_existing=False
):
    """
    Prepare temp_root.

    By default this is non-destructive: if temp_root already exists, it is left
    in place.  Pass clear_existing=True only when you explicitly want to remove
    all previously extracted temp data.
    """
    if os.path.exists(temp_root):

        if clear_existing:

            print(f"Removing existing temp folder: {temp_root}")

            shutil.rmtree(temp_root)

        else:

            print(f"Preserving existing temp folder: {temp_root}")

    os.makedirs(
        temp_root,
        exist_ok=True
    )


def copy_and_untar_raw_data_to_temp(
    date_strings,
    data_sources,
    raw_root=None,
    temp_root="./evaluation/temp",
    keep_tar=False,
    strict=False,
    clear_temp=False,
    skip_existing=True
):
    """
    Copy and untar selected data source/date TARs into temp_root.

    Expected raw layout:

        raw_root/
            cctv/
                20250102.tar
            air_data/
                20250102.tar

    Output layout:

        temp_root/
            cctv/
                20250102/
                    some_data
            air_data/
                20250102/
                    some_data

    Existing temp_root contents are preserved by default.  Set clear_temp=True
    only when you explicitly want to remove the whole temp folder first.

    Parameters
    ----------
    date_strings:
        List of date strings like ["20250102", "20250103"].
    data_sources:
        List of source folder names like ["cctv", "air_data"].
    raw_root:
        Folder containing source folders.
    temp_root:
        Output temp folder.
    keep_tar:
        If True, keep the copied TAR under temp_root/source/date/date.tar.
        If False, delete the copied TAR after extraction.
    strict:
        If True, raise FileNotFoundError when any requested TAR is missing.
        If False, print warnings and continue.
    clear_temp:
        If True, delete temp_root before extraction. Defaults to False.
    skip_existing:
        If True, skip source/date folders that already contain extracted files.
        Defaults to True.
    """
    raw_root = raw_root or configured_raw_root()
    reset_temp_folder(
        temp_root,
        clear_existing=clear_temp
    )

    copied_count = 0
    skipped_existing_count = 0
    missing_paths = []

    for data_source in data_sources:

        source_raw_folder = os.path.join(
            raw_root,
            data_source
        )

        source_temp_folder = os.path.join(
            temp_root,
            data_source
        )

        os.makedirs(
            source_temp_folder,
            exist_ok=True
        )

        for date_str in date_strings:

            if skip_existing and temp_source_date_has_extracted_data(
                temp_root,
                data_source,
                date_str
            ):

                print(
                    f"Skipping existing extracted data: "
                    f"{os.path.join(source_temp_folder, date_str)}"
                )

                skipped_existing_count += 1
                continue

            tar_filename = f"{date_str}.tar"

            source_tar_path = os.path.join(
                source_raw_folder,
                tar_filename
            )

            if not os.path.exists(source_tar_path):

                missing_paths.append(
                    source_tar_path
                )
                continue

            date_temp_folder = os.path.join(
                source_temp_folder,
                date_str
            )

            os.makedirs(
                date_temp_folder,
                exist_ok=True
            )

            copied_tar_path = os.path.join(
                date_temp_folder,
                tar_filename
            )

            print(
                f"Copying: {source_tar_path}"
            )
            print(
                f"  -> {copied_tar_path}"
            )

            shutil.copy2(
                source_tar_path,
                copied_tar_path
            )

            #
            # If the TAR already contains the date folder as its top-level
            # directory, extract into source_temp_folder so the final path is:
            #
            #   temp_root/source/date/...
            #
            # Otherwise, extract directly into date_temp_folder.
            #
            if tar_has_date_top_folder(
                copied_tar_path,
                date_str
            ):

                extract_folder = source_temp_folder

            else:

                extract_folder = date_temp_folder

            print(
                f"Untarring into: {extract_folder}"
            )

            safe_extract_tar(
                copied_tar_path,
                extract_folder
            )

            if not keep_tar:

                os.remove(
                    copied_tar_path
                )

            copied_count += 1

    if missing_paths:

        print("\nWarning: missing requested TAR files:")

        for missing_path in missing_paths:

            print(f"  {missing_path}")

        if strict:

            raise FileNotFoundError(
                "One or more requested TAR files were missing."
            )

    print("\nDone preparing temp data.")
    print(f"Copied/untarred TARs: {copied_count}")
    print(f"Skipped existing extracted folders: {skipped_existing_count}")
    print(f"Temp folder: {temp_root}")



def main():
    """
    Edit these values directly, then run:

        python data_process.py

    This preserves ./evaluation/temp by default, then copies and untars
    missing requested raw TAR files into:

        ./evaluation/temp/<data_source>/<date_str>/...
    """

    date_strings = [
        "20260402",
        # "20250103",
    ]

    data_sources = [
        "cctv",
        "air_data",
        "alertcalifornia",
        "pem_data_station_5min",
        "weather_data",
        "twitter_data",
        "citizen_data",
    ]

    paths = get_config().get("paths", {})
    raw_root = paths.get("raw_archive_root", "./raw_data")
    temp_root = paths.get("evaluation_temp_root", "./evaluation/temp")

    keep_tar = False
    strict = False
    clear_temp = False
    skip_existing = True

    copy_and_untar_raw_data_to_temp(
        date_strings=date_strings,
        data_sources=data_sources,
        raw_root=raw_root,
        temp_root=temp_root,
        keep_tar=keep_tar,
        strict=strict,
        clear_temp=clear_temp,
        skip_existing=skip_existing
    )


if __name__ == "__main__":

    main()
