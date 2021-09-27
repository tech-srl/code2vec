public void actionPerformed(ActionEvent e)
            {
                if (groupCombo.getSelectedItem() != null
                    && groupCombo.getSelectedItem().equals(newGroupString))
                {
                    CreateGroupDialog dialog
                        = new CreateGroupDialog(parentDialog, false);
                    dialog.setModal(true);
                    dialog.setVisible(true);

                    MetaContactGroup newGroup = dialog.getNewMetaGroup();

                    if (newGroup != null)
                    {
                        groupCombo.insertItemAt(newGroup,
                                groupCombo.getItemCount() - 2);
                        groupCombo.setSelectedItem(newGroup);
                    }
                    else
                        groupCombo.setSelectedIndex(0);
                }
            }