# -*- coding: utf-8 -*-

import logging as log
log.basicConfig( format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)04d] %(message)s', datefmt='%Y-%m-%d:%H:%M:%S', level=log.INFO )

import os, glob

class Common :

    def __init__(self):
        pass
    pass

    def open_file_or_folder(self, path) :
        ''' open file or folder by an explorer'''
        import webbrowser as web
        web.open( path )
    pass # -- open_file_or_folder

    def file_name_except_path_ext(self, path):
        # 확장자와 파일 패스를 제외한 파일 이름 구하기.
        head, file_name = os.path.split(path)

        dot_idx = file_name.rfind(".")
        file_name = file_name[: dot_idx]

        return file_name
    pass # file_name_except_path_ext

    def is_writable(self, file):
        # 파일 쓰기 가능 여부 체크
        if os.path.exists(file):
            try:
                os.rename(file, file)

                return True
            except OSError as e:
                log.info( f"is is not writable. {file}" )
            pass
        else :
            return True
        pass

        return False
    pass # -- is_writable

    def to_excel_letter(self, col):
        excel_column = ""

        AZ_len = ord('Z') - ord('A') + 1

        def to_alhapet(num):
            c = chr(ord('A') + int(num))
            return c

        pass  # -- to_alhapet

        while col > 0:
            col, remainder = divmod(col - 1, AZ_len)

            excel_column = to_alhapet(remainder) + excel_column
        pass

        if not excel_column:
            excel_column = "A"
        pass

        return excel_column
    pass  # -- to_excel_letter

    def remove_space_except_first(self, s):
        # 첫 글자를 제외한 나머지 모음을 삭제한다.
        import re
        reg_exp = r'[aeiou]'
        s = s[0] + re.reg_str(reg_exp, '', s[1:])

        return s
    pass # -- remove_space_except_first

    def chdir_to_curr_file(self) :
        # 현재 파일의 폴더로 실행 폴더를 이동함.
        log.info( f"Pwd 1: {os.getcwd()}" )

        dir_name = os.path.dirname(__file__) # change working dir to current file

        if dir_name :
            os.chdir( dir_name )
            log.info(f"Pwd 2: {os.getcwd()}")
        pass
    pass # -- chdir_to_curr_file

    def next_file(self, fileName , debug = False ) :
        directory = os.path.dirname(fileName)
        log.info(f"dir = {directory}")

        _, ext = os.path.splitext(fileName)
        ext = ext.lower()

        find_files = f"{directory}/*{ext}"
        find_files = find_files.replace( "\\", "/" )
        log.info(f"find_files={find_files}")

        files = glob.glob(find_files)

        file_next = None

        fileBaseOrg = os.path.basename(fileName)

        for file in files:
            fileBase = os.path.basename(file)
            debug and log.info(f"fileBase = {fileBase}")

            if fileBase > fileBaseOrg:
                file_next = file
                break
            pass
        pass

        if file_next is not None :
            file_next = file_next.replace("\\", "/")
        pass

        log.info(f"file_next = {file_next}")

        return file_next
    pass # -- next_file

    def prev_file(self, fileName , debug = False ) :
        directory = os.path.dirname(fileName)
        log.info(f"dir = {directory}")

        _, ext = os.path.splitext(fileName)
        ext = ext.lower()

        find_files = f"{directory}/*{ext}"
        find_files = find_files.replace("\\", "/")
        log.info(f"find_files={find_files}")

        files = glob.glob(find_files)

        fileBaseOrg = os.path.basename(fileName)

        file_prev = None

        for file in files:
            fileBase = os.path.basename(file)
            debug and log.info(f"fileBase = {fileBase}")

            if fileBase == fileBaseOrg or fileBase > fileBaseOrg :
                break
            elif fileBase < fileBaseOrg:
                file_prev = file
            else :
                break
            pass
        pass

        if file_prev is not None :
            file_prev = file_prev.replace("\\", "/")
        pass

        log.info(f"file_prev = {file_prev}")

        return file_prev
    pass # -- next_file

    def save_recent_file(self, settings, fileName ) :
        recent_file_list = settings.value('recent_file_list', [], str)

        if fileName in recent_file_list:
            recent_file_list.remove(fileName)
            recent_file_list.insert(0, fileName)
        else:
            recent_file_list.insert(0, fileName)
        pass

        if len(recent_file_list) > 9:
            recent_file_list.pop(len(recent_file_list) - 1)
        pass

        settings.setValue("recent_file_list", recent_file_list)
    pass # save_recent_file

pass