Set xlApp = CreateObject("Excel.Application")
xlApp.Visible = True ' 或设置为 False 以在后台运行

Set xlBook = xlApp.Workbooks.Open("H:\bat_used.xlsm", 0, True)
xlApp.Run "UpdateAndSaveWorkbook"
xlBook.Close False
xlApp.Quit

Set xlBook = Nothing
Set xlApp = Nothing
